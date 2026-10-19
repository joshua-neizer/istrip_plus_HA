"""Support for iStrip+ BLE LED lights."""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from bleak_retry_connector import BleakClientWithServiceCache, establish_connection
from homeassistant.components.bluetooth import (
    BluetoothCallbackMatcher,
    BluetoothChange,
    BluetoothScanningMode,
    BluetoothServiceInfoBleak,
    async_ble_device_from_address,
    async_register_callback,
)
from homeassistant.components.light import (
    ATTR_BRIGHTNESS,
    ATTR_EFFECT,
    ATTR_RGB_COLOR,
    ColorMode,
    LightEntity,
    LightEntityFeature,
)
from homeassistant.config_entries import ConfigEntry
from homeassistant.core import CALLBACK_TYPE, HomeAssistant, callback
from homeassistant.helpers.debounce import Debouncer
from homeassistant.helpers.device_registry import DeviceInfo
from homeassistant.helpers.entity_platform import AddEntitiesCallback
from homeassistant.helpers.event import async_call_later, async_track_time_interval
from homeassistant.helpers.restore_state import ExtraStoredData, RestoreEntity

from .ble_helpers import find_notify_char
from .const import (
    BACKGROUND_CONNECT_ATTEMPTS,
    CONF_KEEP_ALIVE,
    CONF_KEEP_ALIVE_INTERVAL,
    DEFAULT_KEEP_ALIVE,
    DEFAULT_KEEP_ALIVE_INTERVAL,
    DEFAULT_SPEED,
    DOMAIN,
    EFFECT_DEFAULT_SPEEDS,
    KNOWN_CHAR_UUIDS,
    MAX_KEEP_ALIVE_INTERVAL,
    MIN_KEEP_ALIVE_INTERVAL,
    RECONNECT_BACKOFF,
    SPEED_SEND_INTERVAL,
    USER_CONNECT_ATTEMPTS,
)
from .payload_generator import CommandType, PayloadGenerator

_LOGGER = logging.getLogger(__name__)


def _clamp_speed(speed: int) -> int:
    """Clamp to the device's 1-100 range.

    The official app coerces a slider value of 0 up to 1 rather than sending
    it (ClassicEffectFragment: `if (i == 0) i = 1;`).
    """
    return max(1, min(100, speed))


@dataclass
class IstripExtraStoredData(ExtraStoredData):
    """Per-effect speeds, which are not expressible as entity attributes."""

    effect_speeds: dict[str, int]

    def as_dict(self) -> dict[str, Any]:
        """Return a serialisable representation."""
        return {"effect_speeds": self.effect_speeds}


async def async_setup_entry(
    hass: HomeAssistant,
    entry: ConfigEntry,
    async_add_entities: AddEntitiesCallback,
) -> None:
    """Set up iStrip light from config entry."""
    address: str = entry.data["address"]
    char_uuid: str | None = entry.data.get("char_uuid")
    name: str = entry.data.get("name", "iStrip")
    keep_alive: bool = entry.options.get(CONF_KEEP_ALIVE, DEFAULT_KEEP_ALIVE)
    keep_alive_interval = max(
        MIN_KEEP_ALIVE_INTERVAL,
        min(
            MAX_KEEP_ALIVE_INTERVAL,
            int(
                entry.options.get(CONF_KEEP_ALIVE_INTERVAL, DEFAULT_KEEP_ALIVE_INTERVAL)
            ),
        ),
    )
    async_add_entities(
        [
            IstripLight(
                address,
                char_uuid,
                name,
                entry.entry_id,
                keep_alive=keep_alive,
                keep_alive_interval=keep_alive_interval,
            )
        ]
    )


class IstripLight(LightEntity, RestoreEntity):
    """Representation of an iStrip+ BLE LED light."""

    _attr_has_entity_name = True
    _attr_name = None
    _attr_translation_key = "istrip_light"
    _attr_supported_color_modes: set[ColorMode] = {ColorMode.RGB}  # noqa: RUF012
    _attr_color_mode = ColorMode.RGB
    _attr_supported_features = LightEntityFeature.EFFECT

    def __init__(
        self,
        address: str,
        char_uuid: str | None,
        name: str,
        entry_id: str,
        *,
        keep_alive: bool = DEFAULT_KEEP_ALIVE,
        keep_alive_interval: int = DEFAULT_KEEP_ALIVE_INTERVAL,
    ) -> None:
        """Initialize the iStrip light."""
        self._address = address
        self._char_uuid = char_uuid
        self._entry_id = entry_id
        self._pg = PayloadGenerator()
        self._client: BleakClientWithServiceCache | None = None
        self._connected = False

        # Per-effect speeds, seeded from the defaults and overridable at
        # runtime. _pending_speed holds a speed set while no effect was
        # active, to be applied to whichever effect is selected next.
        self._effect_speeds: dict[str, int] = dict(EFFECT_DEFAULT_SPEEDS)
        self._pending_speed: int | None = None
        self._speed_debouncer: Debouncer | None = None
        # Serializes connect/subscribe/join so a concurrent call can't race
        # ahead and send a command before the join handshake has completed.
        self._connect_lock = asyncio.Lock()

        self._keep_alive = keep_alive
        self._keep_alive_interval = keep_alive_interval
        self._keep_alive_running = False
        self._removing = False
        # Pending backoff timer; background reconnects wait while it is set.
        self._reconnect_unsub: CALLBACK_TYPE | None = None
        self._connect_failures = 0
        self._unavailable_logged = False

        mac = address.lower().replace(":", "")
        self._attr_unique_id = f"istrip_{mac}"
        self._attr_is_on = False
        self._attr_rgb_color = (255, 255, 255)
        self._attr_brightness = 255
        self._attr_effect = None
        self._attr_effect_list = list(self._pg.EFFECT_MODES.keys())
        self._attr_device_info = DeviceInfo(
            identifiers={(DOMAIN, address)},
            name=name,
            manufacturer="iStrip",
            model="iStrip+ BLE",
        )

    async def async_turn_on(self, **kwargs: Any) -> None:
        """Turn on the light with optional color, brightness, or effect."""
        if ATTR_EFFECT in kwargs:
            effect_name = kwargs[ATTR_EFFECT]
            self._adopt_pending_speed(effect_name)
            self._attr_effect = effect_name
            if ATTR_BRIGHTNESS in kwargs:
                self._attr_brightness = kwargs[ATTR_BRIGHTNESS]

            self._attr_is_on = True
            await self._send_payload(self._build_effect_payload(effect_name))
            return

        if ATTR_RGB_COLOR in kwargs:
            self._attr_effect = None
            self._attr_rgb_color = kwargs[ATTR_RGB_COLOR]

        r, g, b = self._attr_rgb_color

        if ATTR_BRIGHTNESS in kwargs:
            self._attr_brightness = kwargs[ATTR_BRIGHTNESS]

        # Always convert to the device's 0-100 scale, even when brightness
        # wasn't explicitly passed (e.g. a color-only call while already on).
        brightness = int(self._attr_brightness * 100 / 255)

        if self._attr_effect and ATTR_RGB_COLOR not in kwargs:
            payload = self._build_effect_payload(self._attr_effect)
        else:
            payload = self._pg.get_rgb_payload(r, g, b, brightness)

        self._attr_is_on = True
        await self._send_payload(payload)

    async def async_turn_off(self, **kwargs: Any) -> None:
        """Turn off the light."""
        self._attr_is_on = False
        payload = self._pg.send_led_off()
        await self._send_payload(payload)

    async def set_effect(self, effect_name: str, speed: int | None = None) -> None:
        """Set an effect, optionally overriding that effect's speed."""
        if speed is not None:
            self._store_speed(effect_name, speed)

        self._adopt_pending_speed(effect_name)
        self._attr_effect = effect_name
        payload = self._build_effect_payload(effect_name)

        if self._attr_is_on:
            await self._send_payload(payload)

    async def set_speed(self, speed: int) -> None:
        """Set the speed of the active effect.

        Mirrors the official app, which always stores the value and only
        transmits it when an effect is running: ClassicEffectFragment.
        sendSpeed calls setSpeed unconditionally, then guards the send with
        getMode() > 0. Discarding the value instead would break the natural
        order of setting a speed first and picking an effect second.
        """
        if self._attr_effect is None:
            self._pending_speed = _clamp_speed(speed)
            _LOGGER.debug(
                "No effect active; holding speed %s for the next effect",
                self._pending_speed,
            )
            return

        self._store_speed(self._attr_effect, speed)

        if self._attr_is_on and self._speed_debouncer is not None:
            await self._speed_debouncer.async_call()

    def _speed_for(self, effect_name: str | None) -> int:
        """Return the stored speed for an effect, or the device default."""
        if effect_name is None:
            return DEFAULT_SPEED
        return self._effect_speeds.get(effect_name, DEFAULT_SPEED)

    def _store_speed(self, effect_name: str, speed: int) -> None:
        """Record the speed for a specific effect."""
        self._effect_speeds[effect_name] = _clamp_speed(speed)

    def _adopt_pending_speed(self, effect_name: str) -> None:
        """Apply a speed that was set while no effect was active."""
        if self._pending_speed is None:
            return
        self._effect_speeds[effect_name] = self._pending_speed
        self._pending_speed = None

    def _build_effect_payload(self, effect_name: str) -> str:
        """Build an effect payload at that effect's stored speed."""
        device_brightness = max(10, int(self._attr_brightness * 100 / 255))
        return self._pg.get_effect_payload(
            effect_name,
            device_brightness,
            self._speed_for(effect_name),
            self._attr_rgb_color,
        )

    async def _async_send_active_effect(self) -> None:
        """Resend the active effect. Debounced entry point for speed changes."""
        if self._attr_effect is None or not self._attr_is_on:
            return
        await self._send_payload(self._build_effect_payload(self._attr_effect))

    async def async_added_to_hass(self) -> None:
        """Run when entity is added to Home Assistant."""
        await super().async_added_to_hass()

        # Restore the last known color/brightness/effect across a HA
        # restart, since the device has no reliable way to be asked for
        # its current state (see _handle_notification).
        last_state = await self.async_get_last_state()
        if last_state is not None:
            self._attr_is_on = last_state.state == "on"
            if (rgb := last_state.attributes.get(ATTR_RGB_COLOR)) is not None:
                self._attr_rgb_color = tuple(rgb)
            if (brightness := last_state.attributes.get(ATTR_BRIGHTNESS)) is not None:
                self._attr_brightness = brightness
            if (effect := last_state.attributes.get(ATTR_EFFECT)) is not None:
                self._attr_effect = effect

        # Per-effect speeds cannot be expressed as entity attributes, so they
        # travel as extra stored data. Without this they reset to defaults on
        # every restart, which would be inconsistent with the effect itself
        # being restored just above.
        if (extra := await self.async_get_last_extra_data()) is not None:
            stored = extra.as_dict().get("effect_speeds")
            if isinstance(stored, dict):
                self._effect_speeds.update(
                    {
                        name: _clamp_speed(value)
                        for name, value in stored.items()
                        if isinstance(value, int)
                    }
                )

        # Collapse bursts of speed changes into one write. An automation bound
        # to an input_number can fire far faster than the device can usefully
        # respond; the official app throttles its own slider for this reason.
        self._speed_debouncer = Debouncer(
            self.hass,
            _LOGGER,
            cooldown=SPEED_SEND_INTERVAL,
            immediate=True,
            function=self._async_send_active_effect,
        )

        # The device may not be advertising when Home Assistant starts, in
        # which case no scanner can see it yet and the initial connect below
        # does nothing. Watch for it appearing rather than staying dead until
        # something happens to send a command.
        self.async_on_remove(
            async_register_callback(
                self.hass,
                self._async_device_seen,
                BluetoothCallbackMatcher(address=self._address),
                BluetoothScanningMode.ACTIVE,
            )
        )

        if self._keep_alive:
            _LOGGER.debug(
                "Keep-alive enabled for %s, pinging every %ss",
                self._address,
                self._keep_alive_interval,
            )
            self.async_on_remove(
                async_track_time_interval(
                    self.hass,
                    self._async_keep_alive_tick,
                    timedelta(seconds=self._keep_alive_interval),
                    name=f"iStrip keep-alive {self._address}",
                    cancel_on_shutdown=True,
                )
            )

        await self._ensure_connected()

    @callback
    def _async_device_seen(
        self,
        service_info: BluetoothServiceInfoBleak,
        change: BluetoothChange,
    ) -> None:
        """Connect when a scanner reports the device.

        Skipped while connecting or backing off, so frequent advertisements
        cannot queue up back-to-back attempts.
        """
        if (
            self._connected
            or self._removing
            or self._connect_lock.locked()
            or self._reconnect_unsub is not None
        ):
            return
        self.hass.async_create_task(self._async_background_connect())

    @property
    def extra_restore_state_data(self) -> IstripExtraStoredData:
        """Persist per-effect speeds across restarts."""
        return IstripExtraStoredData(effect_speeds=dict(self._effect_speeds))

    async def async_will_remove_from_hass(self) -> None:
        """Run when entity is removed from Home Assistant."""
        await super().async_will_remove_from_hass()
        self._removing = True
        self._cancel_scheduled_reconnect()
        if self._speed_debouncer is not None:
            self._speed_debouncer.async_shutdown()
        await self._disconnect()

    async def _ensure_connected(
        self, max_attempts: int = USER_CONNECT_ATTEMPTS
    ) -> None:
        """Ensure BLE connection is established and notifications are enabled."""
        async with self._connect_lock:
            if self._connected and self._client and self._client.is_connected:
                return

            try:
                if self._client:
                    await self._disconnect()

                ble_device = async_ble_device_from_address(
                    self.hass, self._address, connectable=True
                )
                if ble_device is None:
                    # Normal at startup: the lamp simply has not advertised
                    # yet. _async_device_seen retries as soon as a scanner
                    # reports it, so this is not an error.
                    # Costs no radio time, so it does not start a backoff.
                    _LOGGER.debug(
                        "No connectable Bluetooth scanner sees %s yet; will "
                        "connect when it is next seen",
                        self._address,
                    )
                    self._mark_unavailable("no Bluetooth scanner can see it")
                    return

                self._client = await establish_connection(
                    BleakClientWithServiceCache,
                    ble_device,
                    self._attr_device_info["name"],
                    disconnected_callback=self._on_disconnected,
                    max_attempts=max_attempts,
                )
                _LOGGER.debug("Connected to iStrip device at %s", self._address)

                # If char_uuid was not discovered during setup, find it now
                if not self._char_uuid:
                    self._char_uuid = self._discover_char_uuid_from_services()
                    if not self._char_uuid:
                        _LOGGER.error(
                            "No writable characteristic found on %s", self._address
                        )
                        await self._disconnect()
                        return
                    _LOGGER.info(
                        "Discovered characteristic UUID %s on %s",
                        self._char_uuid,
                        self._address,
                    )
                    self._persist_char_uuid()

                notify_char_uuid = find_notify_char(self._client, self._char_uuid)
                if notify_char_uuid:
                    try:
                        await self._client.start_notify(
                            notify_char_uuid, self._handle_notification
                        )
                        _LOGGER.debug(
                            "Subscribed to BLE notifications on %s", notify_char_uuid
                        )
                    # bleak/BLE backends raise a wide, inconsistent variety of
                    # errors here; any failure should be logged the same way.
                    except Exception:  # noqa: BLE001
                        _LOGGER.warning(
                            "Could not subscribe to notifications on %s",
                            notify_char_uuid,
                        )
                else:
                    _LOGGER.debug(
                        "No notify-capable characteristic found on %s; "
                        "IR-remote state sync will not be available",
                        self._address,
                    )

                # The device requires a one-time "join group" handshake right
                # after connecting, before it will act on any other command.
                try:
                    join_payload = self._pg.get_join_group_payload()
                    await self._client.write_gatt_char(
                        self._char_uuid, bytes.fromhex(join_payload), response=False
                    )
                    _LOGGER.debug("Sent join-group handshake to %s", self._address)
                # See the notify-subscribe except above: BLE errors here are
                # similarly broad and unpredictable.
                except Exception:  # noqa: BLE001
                    _LOGGER.warning(
                        "Failed to send join-group handshake to %s", self._address
                    )

                # Only mark connected once the join handshake has gone out,
                # so a waiting caller never sends a command ahead of it.
                self._connected = True
                self._mark_available()

            # Connection/discovery can fail in many BLE-stack-specific ways;
            # treat any of them as a failed connection attempt.
            except Exception as err:  # noqa: BLE001
                self._connected = False
                self._client = None
                self._record_connect_failure(err)

    @callback
    def _mark_available(self) -> None:
        """Record a successful connection, logging recovery once."""
        self._attr_available = True
        self._connect_failures = 0
        self._cancel_scheduled_reconnect()
        if self._unavailable_logged:
            _LOGGER.info("iStrip device at %s is available again", self._address)
            self._unavailable_logged = False

    @callback
    def _mark_unavailable(self, reason: str) -> None:
        """Mark the light unavailable, logging it only the first time."""
        self._attr_available = False
        if not self._unavailable_logged:
            _LOGGER.info(
                "iStrip device at %s is unavailable: %s", self._address, reason
            )
            self._unavailable_logged = True

    @callback
    def _record_connect_failure(self, err: Exception) -> None:
        """Mark unavailable and hold off background reconnects for a while."""
        reason = str(err) or type(err).__name__
        self._mark_unavailable(reason)
        self._connect_failures += 1
        delay = RECONNECT_BACKOFF[
            min(self._connect_failures, len(RECONNECT_BACKOFF) - 1)
        ]
        _LOGGER.debug(
            "Connection attempt %d to %s failed (%s); next background attempt in %.0fs",
            self._connect_failures,
            self._address,
            reason,
            delay,
        )
        self._schedule_reconnect(delay)

    def _persist_char_uuid(self) -> None:
        """Store a runtime-discovered characteristic back on the config entry.

        Without this the entry keeps no char_uuid at all after the v3
        migration clears an OTA characteristic, and every connection pays for
        discovery again.
        """
        entry = self.hass.config_entries.async_get_entry(self._entry_id)
        if entry is None or entry.data.get("char_uuid") == self._char_uuid:
            return

        self.hass.config_entries.async_update_entry(
            entry, data={**entry.data, "char_uuid": self._char_uuid}
        )
        _LOGGER.debug(
            "Stored characteristic %s on the config entry for %s",
            self._char_uuid,
            self._address,
        )

    def _discover_char_uuid_from_services(self) -> str | None:
        """Find the best writable characteristic UUID from the connected client."""
        if not self._client or not self._client.is_connected:
            return None

        writable_uuids: list[str] = []
        for service in self._client.services:
            for char in service.characteristics:
                if (
                    "write" in char.properties
                    or "write-without-response" in char.properties
                ):
                    writable_uuids.append(str(char.uuid))

        for known_uuid in KNOWN_CHAR_UUIDS:
            if known_uuid in writable_uuids:
                return known_uuid

        return writable_uuids[0] if writable_uuids else None

    async def _disconnect(self) -> None:
        """Disconnect from the BLE device."""
        client = self._client
        if client:
            # Detach first so _on_disconnected ignores this disconnect.
            self._client = None
            self._connected = False
            try:
                if client.is_connected:
                    await client.disconnect()
                _LOGGER.debug("Disconnected from iStrip device at %s", self._address)
            except Exception:  # noqa: BLE001 - disconnect can fail in many ways
                _LOGGER.error("Error disconnecting from %s", self._address)

    @callback
    def _on_disconnected(self, client: BleakClientWithServiceCache) -> None:
        """Handle the BLE link dropping without us asking it to."""
        if client is not self._client or self._removing:
            # A client we already replaced or deliberately disconnected.
            return

        if not self._keep_alive:
            # Without keep-alive, reconnect lazily on the next command.
            _LOGGER.debug("iStrip device at %s disconnected", self._address)
            return

        _LOGGER.debug(
            "iStrip device at %s disconnected unexpectedly; reconnecting in %.0fs",
            self._address,
            RECONNECT_BACKOFF[0],
        )
        self._connected = False
        self._schedule_reconnect(RECONNECT_BACKOFF[0])

    @callback
    def _schedule_reconnect(self, delay: float) -> None:
        """Start a backoff period, unless one is already pending."""
        if self._removing or self._reconnect_unsub is not None:
            return
        self._reconnect_unsub = async_call_later(
            self.hass, delay, self._async_scheduled_reconnect
        )

    @callback
    def _cancel_scheduled_reconnect(self) -> None:
        """Cancel a pending reconnect, if any."""
        if self._reconnect_unsub is not None:
            self._reconnect_unsub()
            self._reconnect_unsub = None

    async def _async_scheduled_reconnect(self, _now: datetime) -> None:
        """End a backoff period; with keep-alive on, reconnect now."""
        self._reconnect_unsub = None
        if self._keep_alive and not self._connected:
            await self._async_background_connect()

    async def _async_background_connect(self) -> None:
        """Make one connection attempt, unless connecting or backing off."""
        if (
            self._removing
            or self._connect_lock.locked()
            or self._reconnect_unsub is not None
        ):
            return
        was_available = self._attr_available
        _LOGGER.debug("Reconnecting to %s in the background", self._address)
        await self._ensure_connected(max_attempts=BACKGROUND_CONNECT_ATTEMPTS)
        if (
            self._attr_available != was_available
            and self.hass is not None
            and not self._removing
        ):
            self.async_write_ha_state()

    async def _async_keep_alive_tick(self, _now: datetime) -> None:
        """Run the periodic keep-alive."""
        await self._async_keep_alive()

    async def _async_keep_alive(self) -> None:
        """Ping the device if connected, or reconnect if not.

        The ping is the join-group handshake: the device echoes it back without
        changing its output, and _handle_notification ignores the echo.
        """
        if self._removing or self._keep_alive_running:
            return
        if self._connect_lock.locked():
            return

        self._keep_alive_running = True
        try:
            client = self._client
            if not self._connected or client is None or not client.is_connected:
                await self._async_background_connect()
            else:
                try:
                    await client.write_gatt_char(
                        self._char_uuid,
                        bytes.fromhex(self._pg.get_join_group_payload()),
                        response=False,
                    )
                    _LOGGER.debug("Keep-alive ping sent to %s", self._address)
                except Exception as err:  # noqa: BLE001
                    _LOGGER.debug(
                        "Keep-alive ping to %s failed (%s); reconnecting",
                        self._address,
                        err,
                    )
                    self._connected = False
                    await self._async_background_connect()
        finally:
            self._keep_alive_running = False

    def _handle_notification(self, sender: int, data: bytearray) -> None:
        """Handle BLE notification from device."""
        try:
            decrypted = self._pg.decrypt_payload(bytes(data))
            state = self._pg.parse_state(decrypted)

            # The device echoes the join-group handshake back as a
            # notification right after connecting. That echo decodes to
            # R=G=B=0/brightness=0 -- it's not a real state update, so
            # applying it would wrongly show the light as off/black on
            # every reconnect (HA restart, ESP32 reboot, etc.).
            if state["command_type"] != CommandType.RGB:
                return

            self._attr_is_on = state["is_on"]
            self._attr_rgb_color = state["rgb"]
            self._attr_brightness = state["brightness"]
            self._attr_effect = state["effect"]
            # A notification carries the speed of the effect reported in the
            # same frame, so attribute it to that effect rather than to a
            # single global value. This is how a physical IR remote change
            # gets reflected back into the per-effect table.
            if state["effect"] is not None:
                self._store_speed(state["effect"], state["speed"])

            self.schedule_update_ha_state()

        except Exception:
            _LOGGER.exception("Error handling notification from %s", self._address)

    async def _send_payload(
        self, hex_payload: str, retries: int = 2, retry_delay: float = 3.0
    ) -> None:
        """Send a payload to the device via BLE.

        The ESP32 Bluetooth proxy can occasionally crash and reboot
        mid-connection, recovering within a few seconds. Retrying gives it
        a chance to come back before we surface a failure to Home Assistant.
        """
        last_exc: Exception | None = None

        for attempt in range(1, retries + 1):
            try:
                await self._ensure_connected()

                if not self._client or not self._client.is_connected:
                    raise ConnectionError("Not connected to device")

                await self._client.write_gatt_char(
                    self._char_uuid, bytes.fromhex(hex_payload), response=False
                )
                return

            # Write failures span BleakError, timeouts, and ESPHome-proxy-
            # specific errors; any of them means "retry the send".
            except Exception as err:  # noqa: BLE001
                last_exc = err
                self._connected = False
                if attempt < retries:
                    _LOGGER.warning(
                        "Send attempt %d/%d to %s failed (%s), retrying in %.1fs",
                        attempt,
                        retries,
                        self._address,
                        err,
                        retry_delay,
                    )
                    await asyncio.sleep(retry_delay)

        _LOGGER.error(
            "Failed to send BLE payload to %s after %d attempts", self._address, retries
        )
        raise last_exc
