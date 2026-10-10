$ErrorActionPreference = "Stop"
$cpu = Get-CimInstance Win32_Processor | Select-Object -First 1 Name,Manufacturer,NumberOfCores,NumberOfLogicalProcessors,MaxClockSpeed
$os = Get-CimInstance Win32_OperatingSystem | Select-Object Caption,Version,BuildNumber,OSArchitecture,TotalVisibleMemorySize
$drive = (Get-Location).Path.Substring(0,1)
$part = Get-Partition -DriveLetter $drive
$disk = Get-Disk -Number $part.DiskNumber | Select-Object Number,FriendlyName,Manufacturer,Model,BusType,PartitionStyle,Size,FirmwareVersion
$out = [ordered]@{ captured_at=(Get-Date).ToString("o"); working_directory=(Get-Location).Path; processor=$cpu; project_drive=$drive; project_disk=$disk; operating_system=$os }
$out | ConvertTo-Json -Depth 6 | Out-File -Encoding utf8 hardware_exact.json
Write-Host "wrote hardware_exact.json"
