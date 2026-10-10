# Release assets

Create a GitHub release tagged `v1.0.0` and attach the large raw-result archive as:

`RePressDB_JSS_Artifact_Full_Release_V1.0.0.zip`

Generate a SHA-256 checksum before upload.

PowerShell:

```powershell
Get-FileHash RePressDB_JSS_Artifact_Full_Release_V1.0.0.zip -Algorithm SHA256
```

Linux/macOS:

```bash
sha256sum RePressDB_JSS_Artifact_Full_Release_V1.0.0.zip
```

Store the resulting checksum in `RELEASE_ASSET_SHA256.txt` and commit that text file to the repository. The large ZIP itself should remain a release attachment rather than a tracked Git object.
