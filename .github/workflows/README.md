# GitHub Workflows

This directory contains GitHub Actions workflows for the codec5jchain repository.

## Available Workflows

### SBOM Generation (`sbom-generation.yml`)

Generates Software Bill of Materials (SBOM) for the repository to track dependencies and improve supply chain security.

#### Features

- **Multiple Format Support**: Generates SBOM in both SPDX and CycloneDX formats
- **Automated Scanning**: Runs on push, pull requests, and scheduled weekly
- **Artifact Storage**: SBOM reports are stored as artifacts with 90-day retention
- **Security Integration**: Uses Anchore SBOM Action for comprehensive dependency analysis

#### Triggers

- **Push**: Automatic SBOM generation on commits to `main` branch
- **Pull Request**: SBOM analysis on PRs targeting `main`
- **Schedule**: Weekly scans every Sunday at midnight UTC
- **Manual**: Can be triggered manually via workflow_dispatch

#### SBOM Formats

1. **SPDX JSON** (`sbom-spdx.json`): Industry-standard format supported by the Linux Foundation
2. **CycloneDX JSON** (`sbom-cyclonedx.json`): OWASP standard format focused on security use cases

#### Viewing SBOM Results

After workflow completion:
1. Navigate to the **Actions** tab in the repository
2. Click on the latest SBOM Generation workflow run
3. Download the `sbom-reports` artifact
4. Extract and review the JSON files

#### Integration with Security Tools

The generated SBOMs can be used with:
- Dependency tracking tools
- Vulnerability scanners
- License compliance checkers
- Supply chain security platforms

#### Best Practices

- Review SBOM reports regularly for new dependencies
- Keep dependencies up-to-date based on SBOM findings
- Use SBOM data for license compliance verification
- Integrate with vulnerability databases for security monitoring

## Workflow Permissions

All workflows are configured with minimal required permissions:
- `contents: write` - For artifact uploads
- `security-events: write` - For security reporting
- `actions: read` - For workflow metadata

## Related Workflows

- **CodeQL Analysis**: See issue #3 for CodeQL security scanning workflow

## Support

For issues or questions about workflows:
1. Check the workflow run logs in the Actions tab
2. Review this documentation
3. Create an issue with the `workflow` label
