# Exact CPU and SSD metadata

The experiment log captured `Intel Family 6 Model 183`, CPU counts, nominal frequency, memory size, Windows build, and NTFS. It did not retain the processor marketing string or the SSD model.

Run `environment/capture_hardware.bat` on the same computer that produced the formal v1.7.0 timing results. Copy the following fields from `hardware_exact.json` into the manuscript:

- `processor.Name`
- `processor.NumberOfCores`
- `processor.NumberOfLogicalProcessors`
- `project_disk.FriendlyName` or `project_disk.Model`
- `project_disk.BusType`
- `operating_system.BuildNumber`

Do not infer the CPU name from family/model alone. Do not publish disk serial numbers.
