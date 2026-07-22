#!/usr/bin/env python3
"""
PYNQ-Z2 SSH Hardware Report

Purpose:
    Display the PYNQ-Z2 board, Linux, CPU, memory, storage, network,
    FPGA, temperature, and fixed board specifications as terminal tables.

Usage:
    python3 pynq_hardware_report.py

Optional:
    Save the same report to a text file:
    python3 pynq_hardware_report.py | tee pynq_hardware_report.txt

Design:
    - Uses only the Python standard library.
    - Reads Linux /proc and /sys information directly where possible.
    - Marks missing information as "Unavailable" instead of inventing values.
    - Separates measured/detected values from fixed PYNQ-Z2 specifications.
"""

import os
import platform
import shutil
import socket
import subprocess
import textwrap
from datetime import datetime


UNAVAILABLE = "Unavailable"


def read_text(path):
    """Read a Linux information file and remove null bytes used by Device Tree."""
    try:
        with open(path, "rb") as file:
            value = file.read().replace(b"\x00", b"\n").decode("utf-8", errors="replace")
        value = "\n".join(line.strip() for line in value.splitlines() if line.strip())
        return value if value else UNAVAILABLE
    except (OSError, PermissionError):
        return UNAVAILABLE


def first_readable(paths):
    """Return the first readable sysfs/procfs value from equivalent Linux paths."""
    for path in paths:
        value = read_text(path)
        if value != UNAVAILABLE:
            return value
    return UNAVAILABLE


def run_command(command):
    """Run a read-only system command and return its output without opening a shell."""
    try:
        completed = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False,
        )
        output = completed.stdout.strip()
        return output if completed.returncode == 0 and output else UNAVAILABLE
    except (OSError, ValueError):
        return UNAVAILABLE


def human_bytes(byte_count):
    """Convert a byte count into a compact binary unit for terminal readability."""
    value = float(byte_count)
    units = ("B", "KiB", "MiB", "GiB", "TiB")
    for unit in units:
        if abs(value) < 1024.0 or unit == units[-1]:
            return "{:.2f} {}".format(value, unit)
        value /= 1024.0
    return "{} B".format(byte_count)


def print_table(title, rows):
    """
    Print a two-column ASCII table.

    The value column is wrapped according to the current terminal width so the
    table remains readable in MobaXterm without requiring third-party packages.
    """
    terminal_width = shutil.get_terminal_size((120, 30)).columns
    table_width = max(72, min(terminal_width, 140))
    item_width = min(34, max(24, table_width // 3))
    value_width = table_width - item_width - 7

    border = "+-" + ("-" * item_width) + "-+-" + ("-" * value_width) + "-+"
    print("\n{}".format(title))
    print(border)
    print("| {item:<{iw}} | {value:<{vw}} |".format(
        item="Item", value="Value", iw=item_width, vw=value_width
    ))
    print(border)

    for item, value in rows:
        item_lines = textwrap.wrap(str(item), item_width) or [""]
        value_lines = []
        for original_line in str(value).splitlines() or [""]:
            value_lines.extend(
                textwrap.wrap(
                    original_line,
                    value_width,
                    replace_whitespace=False,
                    drop_whitespace=False,
                ) or [""]
            )

        line_count = max(len(item_lines), len(value_lines))
        for index in range(line_count):
            item_part = item_lines[index] if index < len(item_lines) else ""
            value_part = value_lines[index] if index < len(value_lines) else ""
            print("| {item:<{iw}} | {value:<{vw}} |".format(
                item=item_part,
                value=value_part,
                iw=item_width,
                vw=value_width,
            ))
        print(border)


# ---------------------------------------------------------------------------
# Report header
# ---------------------------------------------------------------------------
print("PYNQ-Z2 HARDWARE AND SYSTEM REPORT")
print("Generated: {}".format(datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
print("Hostname:  {}".format(socket.gethostname()))

# ---------------------------------------------------------------------------
# Board and Device Tree information
# ---------------------------------------------------------------------------
board_model = first_readable([
    "/sys/firmware/devicetree/base/model",
    "/proc/device-tree/model",
])

compatible = first_readable([
    "/sys/firmware/devicetree/base/compatible",
    "/proc/device-tree/compatible",
])

serial_number = first_readable([
    "/sys/firmware/devicetree/base/serial-number",
    "/proc/device-tree/serial-number",
])

print_table("1. BOARD AND SOC DETECTION", [
    ("Detected board model", board_model),
    ("Device Tree compatible", compatible.replace("\n", ", ")),
    ("Device Tree serial number", serial_number),
    ("Machine architecture", platform.machine()),
])

# ---------------------------------------------------------------------------
# Operating system and Linux kernel
# ---------------------------------------------------------------------------
os_release_text = read_text("/etc/os-release")
os_release = {}
if os_release_text != UNAVAILABLE:
    for line in os_release_text.splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            os_release[key] = value.strip().strip('"')

print_table("2. OPERATING SYSTEM", [
    ("Operating system", os_release.get("PRETTY_NAME", os_release.get("NAME", UNAVAILABLE))),
    ("OS version", os_release.get("VERSION", UNAVAILABLE)),
    ("Linux kernel", platform.release()),
    ("Kernel build", platform.version()),
    ("System platform", platform.platform()),
    ("Python version", platform.python_version()),
])

# ---------------------------------------------------------------------------
# CPU details from /proc, sysfs, and the Python platform module
# ---------------------------------------------------------------------------
cpuinfo_text = read_text("/proc/cpuinfo")
cpu_fields = {}
processor_count = 0

if cpuinfo_text != UNAVAILABLE:
    for line in cpuinfo_text.splitlines():
        if ":" not in line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        if key == "processor":
            processor_count += 1
        if key not in cpu_fields and value:
            cpu_fields[key] = value

if processor_count == 0:
    processor_count = os.cpu_count() or 0

cpu_rows = [
    ("Logical CPU cores", processor_count),
    ("CPU model", cpu_fields.get("model name", cpu_fields.get("Processor", UNAVAILABLE))),
    ("Hardware identifier", cpu_fields.get("Hardware", UNAVAILABLE)),
    ("CPU revision", cpu_fields.get("Revision", UNAVAILABLE)),
    ("CPU serial", cpu_fields.get("Serial", UNAVAILABLE)),
    ("BogoMIPS", cpu_fields.get("BogoMIPS", UNAVAILABLE)),
    ("CPU features", cpu_fields.get("Features", UNAVAILABLE)),
]

for cpu_index in range(processor_count):
    current_khz = first_readable([
        "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_cur_freq".format(cpu_index),
        "/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_cur_freq".format(cpu_index),
    ])
    maximum_khz = first_readable([
        "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_max_freq".format(cpu_index),
        "/sys/devices/system/cpu/cpu{}/cpufreq/cpuinfo_max_freq".format(cpu_index),
    ])

    if current_khz != UNAVAILABLE:
        try:
            current_khz = "{:.2f} MHz".format(float(current_khz) / 1000.0)
        except ValueError:
            pass

    if maximum_khz != UNAVAILABLE:
        try:
            maximum_khz = "{:.2f} MHz".format(float(maximum_khz) / 1000.0)
        except ValueError:
            pass

    cpu_rows.append(("CPU{} current frequency".format(cpu_index), current_khz))
    cpu_rows.append(("CPU{} maximum frequency".format(cpu_index), maximum_khz))

print_table("3. CPU DETAILS", cpu_rows)

# ---------------------------------------------------------------------------
# Memory information from /proc/meminfo
# ---------------------------------------------------------------------------
meminfo_text = read_text("/proc/meminfo")
meminfo_kib = {}

if meminfo_text != UNAVAILABLE:
    for line in meminfo_text.splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        parts = raw_value.strip().split()
        if parts and parts[0].isdigit():
            meminfo_kib[key] = int(parts[0])

memory_rows = []
for label, key in [
    ("Linux-visible total memory", "MemTotal"),
    ("Available memory", "MemAvailable"),
    ("Free memory", "MemFree"),
    ("Buffers", "Buffers"),
    ("Cached memory", "Cached"),
    ("Swap total", "SwapTotal"),
    ("Swap free", "SwapFree"),
]:
    if key in meminfo_kib:
        memory_rows.append((label, human_bytes(meminfo_kib[key] * 1024)))
    else:
        memory_rows.append((label, UNAVAILABLE))

if "MemTotal" in meminfo_kib and "MemAvailable" in meminfo_kib:
    used_memory = meminfo_kib["MemTotal"] - meminfo_kib["MemAvailable"]
    memory_rows.insert(1, ("Estimated used memory", human_bytes(used_memory * 1024)))

print_table("4. MEMORY", memory_rows)

# ---------------------------------------------------------------------------
# Storage information from the root filesystem and MicroSD sysfs entries
# ---------------------------------------------------------------------------
root_usage = shutil.disk_usage("/")
root_percent = (root_usage.used / root_usage.total * 100.0) if root_usage.total else 0.0

mmc_sectors = first_readable(["/sys/block/mmcblk0/size"])
if mmc_sectors != UNAVAILABLE:
    try:
        mmc_capacity = human_bytes(int(mmc_sectors) * 512)
    except ValueError:
        mmc_capacity = mmc_sectors
else:
    mmc_capacity = UNAVAILABLE

mount_summary = run_command(["df", "-hP"])
block_summary = run_command(["lsblk", "-o", "NAME,SIZE,TYPE,FSTYPE,MOUNTPOINT"])

print_table("5. STORAGE", [
    ("Root filesystem total", human_bytes(root_usage.total)),
    ("Root filesystem used", human_bytes(root_usage.used)),
    ("Root filesystem free", human_bytes(root_usage.free)),
    ("Root filesystem usage", "{:.1f}%".format(root_percent)),
    ("MicroSD raw capacity", mmc_capacity),
    ("MicroSD device name", first_readable(["/sys/block/mmcblk0/device/name"])),
    ("MicroSD CID", first_readable(["/sys/block/mmcblk0/device/cid"])),
    ("Mounted filesystems", mount_summary),
    ("Block devices", block_summary),
])

# ---------------------------------------------------------------------------
# Network details for every detected non-loopback interface
# ---------------------------------------------------------------------------
network_rows = []
network_root = "/sys/class/net"

if os.path.isdir(network_root):
    interfaces = sorted(
        name for name in os.listdir(network_root)
        if name != "lo"
    )
else:
    interfaces = []

if not interfaces:
    network_rows.append(("Network interfaces", UNAVAILABLE))

for interface in interfaces:
    address_output = run_command([
        "ip", "-4", "-o", "addr", "show", "dev", interface
    ])

    ipv4_addresses = []
    if address_output != UNAVAILABLE:
        for line in address_output.splitlines():
            parts = line.split()
            if "inet" in parts:
                inet_index = parts.index("inet")
                if inet_index + 1 < len(parts):
                    ipv4_addresses.append(parts[inet_index + 1])

    network_rows.extend([
        ("{} IPv4 address".format(interface),
         ", ".join(ipv4_addresses) if ipv4_addresses else UNAVAILABLE),
        ("{} MAC address".format(interface),
         read_text("/sys/class/net/{}/address".format(interface))),
        ("{} operational state".format(interface),
         read_text("/sys/class/net/{}/operstate".format(interface))),
        ("{} carrier".format(interface),
         read_text("/sys/class/net/{}/carrier".format(interface))),
        ("{} link speed".format(interface),
         "{} Mbps".format(read_text("/sys/class/net/{}/speed".format(interface)))
         if read_text("/sys/class/net/{}/speed".format(interface)) != UNAVAILABLE
         else UNAVAILABLE),
    ])

print_table("6. NETWORK", network_rows)

# ---------------------------------------------------------------------------
# USB devices visible to Linux
# ---------------------------------------------------------------------------
lsusb_output = run_command(["lsusb"])
usb_rows = []

if lsusb_output == UNAVAILABLE:
    usb_rows.append(("Detected USB devices", UNAVAILABLE))
else:
    usb_lines = lsusb_output.splitlines()
    usb_rows.append(("USB device count", len(usb_lines)))
    for index, device in enumerate(usb_lines, start=1):
        usb_rows.append(("USB device {}".format(index), device))

print_table("7. USB DEVICES", usb_rows)

# ---------------------------------------------------------------------------
# FPGA status and currently loaded PYNQ overlay
# ---------------------------------------------------------------------------
fpga_state = first_readable([
    "/sys/class/fpga_manager/fpga0/state",
    "/sys/class/fpga_manager/fpga0/status",
])

overlay_bitfile = UNAVAILABLE
overlay_ip_names = UNAVAILABLE

try:
    from pynq import PL

    overlay_bitfile = getattr(PL, "bitfile_name", None) or UNAVAILABLE
    ip_dictionary = getattr(PL, "ip_dict", None)
    if isinstance(ip_dictionary, dict):
        overlay_ip_names = ", ".join(sorted(ip_dictionary.keys())) if ip_dictionary else "No IP entries"
except (ImportError, AttributeError, RuntimeError, OSError):
    # Reporting continues because the system details are still valid even when
    # the current Python interpreter does not have access to the PYNQ package.
    pass

print_table("8. FPGA AND PYNQ OVERLAY", [
    ("FPGA manager state", fpga_state),
    ("Loaded overlay bitfile", overlay_bitfile),
    ("Overlay IP blocks", overlay_ip_names),
])

# ---------------------------------------------------------------------------
# Temperature, uptime, and load
# ---------------------------------------------------------------------------
temperature_raw = first_readable([
    "/sys/class/thermal/thermal_zone0/temp",
])

if temperature_raw != UNAVAILABLE:
    try:
        temperature_value = float(temperature_raw)
        if temperature_value > 1000:
            temperature_value /= 1000.0
        temperature_text = "{:.2f} °C".format(temperature_value)
    except ValueError:
        temperature_text = temperature_raw
else:
    temperature_text = UNAVAILABLE

load_average = read_text("/proc/loadavg")
uptime_seconds_raw = read_text("/proc/uptime")

if uptime_seconds_raw != UNAVAILABLE:
    try:
        uptime_seconds = int(float(uptime_seconds_raw.split()[0]))
        days, remainder = divmod(uptime_seconds, 86400)
        hours, remainder = divmod(remainder, 3600)
        minutes, seconds = divmod(remainder, 60)
        uptime_text = "{} days, {:02d}:{:02d}:{:02d}".format(
            days, hours, minutes, seconds
        )
    except (ValueError, IndexError):
        uptime_text = uptime_seconds_raw
else:
    uptime_text = UNAVAILABLE

print_table("9. HEALTH AND RUNTIME", [
    ("Thermal zone type", first_readable(["/sys/class/thermal/thermal_zone0/type"])),
    ("Board/SoC temperature", temperature_text),
    ("System uptime", uptime_text),
    ("Load average", load_average),
])

# ---------------------------------------------------------------------------
# Fixed PYNQ-Z2 specifications
# These values are reference specifications, not measurements read from Linux.
# ---------------------------------------------------------------------------
print_table("10. PYNQ-Z2 FIXED BOARD SPECIFICATIONS (REFERENCE)", [
    ("SoC", "Xilinx Zynq-7000 XC7Z020"),
    ("Processing system", "Dual-core ARM Cortex-A9"),
    ("Installed DDR3 memory", "512 MB"),
    ("FPGA logic cells", "85,000"),
    ("FPGA lookup tables", "53,200 LUTs"),
    ("FPGA flip-flops", "106,400"),
    ("FPGA Block RAM", "630 KB"),
    ("FPGA DSP slices", "220"),
    ("Quad-SPI flash", "128 Mbit"),
    ("Ethernet capability", "10/100/1000 Mbps"),
    ("USB capability", "USB-UART/JTAG and USB 2.0 OTG"),
    ("Measurement note",
     "This section contains fixed board specifications. "
     "Use Vivado reports for implemented LUT, FF, BRAM, DSP, and timing usage."),
])

print("\nReport complete.")
