# Testing the Security Notification Workflow

This document describes how to test the security vulnerability notification system.

## Test Scenarios

### Scenario 1: Pull Request from Forked Repository

**Setup:**
1. Fork the repository to a different GitHub account
2. Make changes in the forked repository
3. Create a pull request from the fork to the main repository

**Expected Behavior:**
- Workflow triggers on `pull_request` or `pull_request_target` event
- Detects that `github.event.pull_request.head.repo.fork == true`
- Generates security alert in workflow logs
- Creates a security issue with label `security`, `alert`, `forked-activity`
- Posts a warning comment on the PR with security checklist

**Verification:**
- Check workflow run logs for security alerts
- Verify security issue was created
- Confirm PR has security warning comment

### Scenario 2: Pull Request from External Repository

**Setup:**
1. Someone creates a PR where the head repository differs from base
2. This happens naturally with fork PRs

**Expected Behavior:**
- Workflow detects external repository source
- Logs show repository names (head vs base)
- Security notifications are generated

**Verification:**
- Review workflow logs for external repository detection
- Check that PR details show correct repository names

### Scenario 3: Push Event to Branch

**Setup:**
1. Push commits to any branch in the repository
2. Can be from local development or remote push

**Expected Behavior:**
- Workflow triggers on `push` event
- Logs commit information (SHA, author, message)
- Monitors for suspicious patterns

**Verification:**
- Check workflow runs after push events
- Verify commit details are logged correctly

### Scenario 4: No Fork Activity (Normal Operation)

**Setup:**
1. Create a PR from a branch within the same repository
2. Or push directly to a branch

**Expected Behavior:**
- Workflow runs but no alerts generated
- `IS_FORK` and `IS_EXTERNAL` remain false
- No security issues created
- No PR comments posted

**Verification:**
- Workflow completes successfully
- No security notifications generated

## Manual Testing Steps

### Test 1: Validate Workflow Syntax

```bash
# Install yamllint
pip install yamllint

# Validate the workflow file
yamllint .github/workflows/security-notification.yml
```

### Test 2: Simulate Fork Detection

You can test the detection logic locally:

```bash
# Set environment variables to simulate a fork PR
export GITHUB_EVENT_NAME="pull_request"
export GITHUB_EVENT_PULL_REQUEST_HEAD_REPO_FORK="true"
export GITHUB_EVENT_PULL_REQUEST_HEAD_REPO_FULL_NAME="external/repo"
export GITHUB_REPOSITORY="kadSerrCX/dump-fullstack"

# Then the script logic would detect this as a fork
```

### Test 3: Review Workflow Permissions

Ensure the workflow has necessary permissions:
- `contents: read` - for checking out code
- `issues: write` - for creating security issues
- `pull-requests: write` - for posting comments

### Test 4: Check Action Versions

Verify that used actions are up to date:
- `actions/checkout@v4` ✓
- `actions/github-script@v7` ✓

## Automated Testing

### CI/CD Integration

The workflow will automatically run on:
- Every push to any branch
- Every PR opened, synchronized, or reopened
- PR target events from forks

### Monitoring

Monitor the workflow by:
1. Going to the Actions tab in GitHub
2. Selecting "Security Vulnerability Notification" workflow
3. Reviewing recent runs

### Debugging

If the workflow fails:
1. Check workflow run logs for error messages
2. Verify YAML syntax is correct
3. Ensure GitHub token has required permissions
4. Check that action versions are compatible

## Expected Outputs

### When Fork Detected

**Workflow Logs:**
```
⚠️ SECURITY ALERT: Pull request is from a forked repository!
⚠️ SECURITY ALERT: Pull request is from external repository!
Source: external/repo
Target: kadSerrCX/dump-fullstack
```

**Security Issue Created:**
- Title: "🔒 Security Alert: Activity from Forked/Remote Source"
- Labels: security, alert, forked-activity
- Contains: PR details, security recommendations, checklist

**PR Comment:**
- Warning header with 🔒 icon
- Source repository information
- Security checklist for reviewers
- Link to workflow run

### When No Fork Detected

**Workflow Logs:**
```
Starting security check for forked/remote activity...
No fork or external activity detected.
```

**No Issues/Comments:**
- No security issues created
- No PR comments posted
- Workflow completes silently

## Performance Considerations

- Workflow runs on every push/PR (be aware of action minutes)
- Uses standard GitHub-hosted runners (no special requirements)
- Minimal resource usage (mainly shell scripts)
- Quick execution time (< 1 minute typically)

## Security Notes

- Workflow uses `GITHUB_TOKEN` (automatically provided)
- No secrets or credentials are exposed
- All operations are read-only except for issue/comment creation
- Uses official GitHub actions only

## Troubleshooting

### Issue: Workflow not triggering

**Solution:**
- Check that workflow file is in `.github/workflows/`
- Verify YAML syntax is valid
- Ensure branch names in triggers are correct

### Issue: Cannot create issues

**Solution:**
- Check repository settings allow issues
- Verify `GITHUB_TOKEN` has `issues: write` permission
- Confirm issues are not disabled for the repository

### Issue: Cannot post PR comments

**Solution:**
- Verify `GITHUB_TOKEN` has `pull-requests: write` permission
- Check that PR is not from a fork with restricted permissions
- Review repository security settings

## Future Enhancements

Potential improvements:
- Integration with external security scanning tools
- Whitelist of trusted contributors
- Customizable notification channels (Slack, email, etc.)
- More sophisticated pattern detection
- Integration with code scanning results
