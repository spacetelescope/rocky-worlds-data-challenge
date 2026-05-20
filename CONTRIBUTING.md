Please open a new issue or new pull request for bugs, feedback, or new features you would like to see. If there is an issue you would like to work on, please leave a comment and we will be happy to assist. New contributions and contributors are very welcome!

New to GitHub or open source projects? If you are unsure about where to start or have not used GitHub before, please feel free to contact the package maintainers.

Feedback and feature requests? Is there something missing you would like to see? Please open an issue or send an email to the maintainers. This package follows the Spacetelescope [Code of Conduct](CODE_OF_CONDUCT.md) and strives to provide a welcoming community to all of our users and contributors.

## Maintainer release process

Package publishing is automated through GitHub Actions in
`.github/workflows/publish_pypi.yml`.

Normal contributors do not need PyPI or TestPyPI access. Maintainers should use PyPI Trusted Publishing / OIDC only; do not create or store long-lived PyPI API tokens in GitHub secrets.

For the public repository, Trusted Publishers have been configured on TestPyPI and PyPI with:

- PyPI project name: `rocky-worlds-data-challenge`
- Owner: `spacetelescope`
- Repository: `rocky-worlds-data-challenge`
- Workflow filename: `publish_pypi.yml`
- TestPyPI environment: `testpypi`
- PyPI environment: `pypi`

Release flow:

1. Merge the release-ready changes to `main`.
2. Create and push an annotated version tag, for example:

   ```bash
   git tag -a v1.0 -m "v1.0"
   git push origin v1.0
   ```

   Pushing a `v*` tag builds the distributions and publishes them to TestPyPI.

3. Verify the TestPyPI upload and install.
4. Create a GitHub Release from the same tag.
   Publishing the GitHub Release builds the distributions again and publishes them to PyPI.

The GitHub environment `pypi` has been configured with required reviewers in the repository settings, so real PyPI publication requires manual approval. The `testpypi` environment does not require manual approval because it is used as the automated release rehearsal when a `v*` tag is pushed.
