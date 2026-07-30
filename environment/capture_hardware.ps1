$ErrorActionPreference = "Stop"

$cpu = Get-CimInstance Win32_Processor | Select-Object Name, Manufacturer, NumberOfCores, NumberOfLogicalProcessors, MaxClockSpeed
$os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption, Version, BuildNumber, OSArchitecture, TotalVisibleMemorySize

$driveLetter = (Get-Location).Drive.Name
$disk = $null
if ($driveLetter) {
    try {
        $disk = Get-Partition -DriveLetter $driveLetter | Get-Disk | Select-Object Number, FriendlyName, Manufacturer, Model, BusType, PartitionStyle, Size, FirmwareVersion
    } catch {
        $disk = Get-CimInstance Win32_DiskDrive | Select-Object Model, Manufacturer, InterfaceType, MediaType, FirmwareRevision, Size
    }
}

$report = [ordered]@{
    captured_at = (Get-Date).ToString("o")
    working_directory = (Get-Location).Path
    processor = $cpu
    project_drive = $driveLetter
    project_disk = $disk
    operating_system = $os
}

$report | ConvertTo-Json -Depth 6 | Set-Content -Encoding UTF8 hardware_exact.json
$report | ConvertTo-Json -Depth 6
Write-Host "Saved: $((Get-Location).Path)\hardware_exact.json"
