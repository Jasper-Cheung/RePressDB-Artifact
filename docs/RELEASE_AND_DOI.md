# GitHub release and archival DOI

1. Create a public GitHub repository named `RePressDB-JSS-Artifact`.
2. Push the contents of the repository ZIP. Do not commit the large full-release ZIP to Git history.
3. Replace `REPLACE-WITH-ACCOUNT` in `CITATION.cff` and in the manuscript link file.
4. Create tag `v1.0.0` and a GitHub Release.
5. Attach `RePressDB_JSS_Artifact_Full_Release_V1.0.0.zip` to that release.
6. Upload the same full-release ZIP to Zenodo, or enable the GitHub repository in Zenodo before publishing a release.
7. Reserve or publish the Zenodo DOI, then replace the DOI placeholder in the manuscript.
8. Verify the release asset against `release_assets/RELEASE_ASSET_SHA256.txt`.

A manual Zenodo upload of the full-release ZIP is the safest way to ensure that the raw latency arrays are included in the DOI-bearing record.
