# GitHub Actions Workflows

This directory contains automated workflows for the repository.

## Available Workflows

### 1. Security Vulnerability Notification (`security-notification.yml`)

**Purpose**: Automatically detects and notifies when commits or pull requests originate from forked or remote repositories.

**Triggers**:
- Push events to any branch
- Pull request events (opened, synchronize, reopened)
- Pull request target events (for fork PRs)

**Actions**:
- Detects if a PR comes from a forked repository
- Identifies external/remote source activity
- Creates security issues with detailed information
- Posts warning comments on pull requests
- Provides security checklist for reviewers

**Benefits**:
- Enhanced security monitoring
- Early detection of potentially risky contributions
- Automated security documentation
- Clear visibility for repository maintainers

### 2. Whitelist Glob SC (`blank.yml`)

**Purpose**: Basic workflow for repository automation tasks.

## Security Considerations

All workflows in this repository follow security best practices:
- Limited permissions (read-only by default)
- Explicit permission grants when needed
- Secure handling of secrets
- Automated security checks

## Adding New Workflows

When adding new workflows:
1. Follow the existing naming conventions
2. Include clear documentation
3. Set appropriate permissions
4. Test thoroughly before merging
5. Add security checks when handling external input

## Testing Workflows

Workflows can be tested using:
- GitHub's workflow validator
- Local tools like `yamllint`
- Act (for local workflow execution)
- GitHub Actions workflow run logs

## Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax Reference](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Security Hardening](https://docs.github.com/en/actions/security-guides/security-hardening-for-github-actions)
