# Security Policy

## Security Vulnerability Notification

This repository includes an automated security notification system that monitors for potentially risky activity from forked or remote sources.

### How It Works

The security notification workflow automatically:

1. **Detects Forked Repository Activity**: Monitors all pull requests to identify if they originate from forked repositories
2. **Identifies External Sources**: Checks if commits or pull requests come from external/remote URLs
3. **Generates Alerts**: Creates detailed security notifications when suspicious activity is detected
4. **Provides Visibility**: Posts comments on pull requests and creates issues for tracking

### Monitored Events

The security system monitors:

- **Push Events**: Commits pushed to any branch
- **Pull Requests**: Pull requests opened, synchronized, or reopened
- **Pull Request Target**: Special handling for pull requests from forks

### Security Notifications

When activity from a forked or remote source is detected, the system:

1. **Logs detailed information** about the event in the workflow run
2. **Creates a security issue** with comprehensive details and recommendations
3. **Posts a comment** on pull requests with a security checklist
4. **Labels the issue** with `security`, `alert`, and `forked-activity` for easy filtering

### Security Checklist for Reviewers

When reviewing pull requests from forked repositories:

- [ ] Verify the identity of the contributor
- [ ] Review all code changes carefully for security issues
- [ ] Check for exposed secrets or credentials
- [ ] Validate changes against project guidelines
- [ ] Run security scans on the proposed changes
- [ ] Test the changes thoroughly in an isolated environment
- [ ] Ensure no malicious code patterns are present

### Workflow Configuration

The security notification workflow is located at:
```
.github/workflows/security-notification.yml
```

### Best Practices

1. **Always review** pull requests from forks with extra scrutiny
2. **Never merge** without thorough code review
3. **Run security scans** before accepting external contributions
4. **Verify contributor identity** when possible
5. **Check commit history** for suspicious patterns
6. **Test changes** in an isolated environment first

### Reporting Security Issues

If you discover a security vulnerability, please report it by:

1. Creating a private security advisory
2. Emailing the repository maintainers
3. Using GitHub's security reporting features

**Do not** create public issues for security vulnerabilities.

### Workflow Permissions

The security workflow requires the following permissions:
- `contents: read` - To checkout the repository
- `issues: write` - To create security notification issues
- `pull-requests: write` - To comment on pull requests

### Customization

You can customize the security notification workflow by:

1. Modifying trigger conditions in the `on:` section
2. Adding custom detection logic in the `Detect Fork or Remote Source` step
3. Adjusting notification formats in the issue creation step
4. Adding integration with external security tools

### Additional Resources

- [GitHub Security Best Practices](https://docs.github.com/en/code-security)
- [Securing Your Repository](https://docs.github.com/en/repositories/managing-your-repositorys-settings-and-features/enabling-features-for-your-repository/managing-security-and-analysis-settings-for-your-repository)
- [About Code Scanning](https://docs.github.com/en/code-security/code-scanning/automatically-scanning-your-code-for-vulnerabilities-and-errors/about-code-scanning)
