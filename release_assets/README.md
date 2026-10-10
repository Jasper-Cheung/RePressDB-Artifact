# Release assets

Create a GitHub release tagged `v1.0.0` and attach the large raw-result archive as:

`RePressDB_JSS_Artifact_Full_Release_V1.0.0.zip`

The archive contains the original v1.5.1 core-results and latency-arrays
archives and the original v1.7.0 core and latency-arrays archives. GitHub
provides source archives for the release tag separately. The two v1.5.1
archives together cover every file in `result_file_manifest.csv`; the two
v1.7.0 archives together cover every
file in `v170_result_manifest.csv`. `RELEASE_CONTENTS.txt` and
`SOURCE_ARCHIVES_SHA256.txt` inside the release archive record its contents
and source-archive checksums. Public source datasets are not included.

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
