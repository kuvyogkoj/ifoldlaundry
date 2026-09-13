
from pyorbbecsdk import (  # type: ignore  # compiled extension; stubs in stubs/pyorbbecsdk.pyi
    Context,
    OBError,
    OBSensorType,
    Pipeline,
)


def main():
    ctx = Context()
    device_list = ctx.query_devices()
    if device_list.get_count() == 0:
        print("No orbec device found - exitting")
        raise SystemExit(1)
    print("Found orbec device ... ")

    for i in range(device_list.get_count()):
        # get_device_by_index() opens the device exclusively.
        # Only one process should open a device at a time.
        device = device_list.get_device_by_index(i)

        info = device.get_device_info()
        print(f"Device #{i + 1}")
        print(f"  Name           : {info.get_name()}")
        print(f"  Serial Number  : {info.get_serial_number()}")
        print(f"  Firmware       : {info.get_firmware_version()}")
        print(f"  Hardware       : {info.get_hardware_version()}")
        print(f"  USB PID / VID  : 0x{info.get_pid():04X} / 0x{info.get_vid():04X}")
        print(f"  Connection     : {info.get_connection_type()}")
        print()

    pipeline = Pipeline(device)

    VIDEO_SENSORS = [
        (OBSensorType.DEPTH_SENSOR, "Depth"),
        (OBSensorType.COLOR_SENSOR, "Color"),
    ]
    print("  Default stream configurations:")
    for sensor_type, label in VIDEO_SENSORS:
        try:
            profiles = pipeline.get_stream_profile_list(sensor_type)
            p = profiles.get_default_video_stream_profile()
            print(
                f"    {label:<10} : {p.get_width()}x{p.get_height()} " f"@ {p.get_fps()} fps  format={p.get_format()}"
            )
        except OBError:
            pass  # sensor not present on this device

        print()
    print("  Depth preset:")
    try:
        current = device.get_current_preset_name()
        print(f"    Active preset  : {current}")
        preset_list = device.get_available_preset_list()
        names = [preset_list[j] for j in range(len(preset_list))]
        print(f"    Available      : {', '.join(names)}")
    except OBError:
        print("    (preset not supported on this device)")

    print()


if __name__ == "__main__":
    main()