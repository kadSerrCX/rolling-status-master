# GitHub Actions Workflows

## CodeQL Analysis Workflow

### Overview
The `codeql-analysis.yml` workflow implements automated security scanning using CodeQL on self-hosted runners with support for JDK runtime analysis.

### Features

#### Self-Hosted Runner Configuration
- Runs on self-hosted infrastructure for better control and performance
- Timeout set to 360 minutes for comprehensive analysis
- Supports both Java and C++ codebases

#### JDK Runtime Environment
- **Distribution**: Eclipse Temurin (formerly AdoptOpenJDK)
- **Version**: Java 17 (LTS)
- **Build Tools**: Maven and Gradle support with automatic detection
- **Caching**: Maven dependencies are cached for faster builds

#### CodeQL CLI Database Creation
The workflow creates CodeQL databases through the following process:
1. **Checkout**: Repository code is checked out with full history
2. **Environment Setup**: JDK runtime is configured for Java projects
3. **CodeQL Initialization**: CodeQL CLI initializes databases for specified languages
4. **Build Process**: Projects are built to populate the database
5. **Analysis**: Security and quality queries are executed
6. **Upload**: Results are uploaded to GitHub Security tab

#### Notification System
The workflow includes notification steps that output:
- **Initialization**: CodeQL database creation start notification
- **Runner Information**: Self-hosted runner identification
- **Language Details**: Current language being analyzed
- **JDK Runtime**: Java version information
- **Completion Status**: Analysis completion with status

### Triggers
- **Push**: Triggers on pushes to the `main` branch
- **Pull Request**: Triggers on pull requests targeting `main`
- **Schedule**: Weekly runs every Sunday at midnight (UTC)

### Languages Analyzed
- **Java**: Full JDK runtime analysis with Maven/Gradle build support
- **C++**: Native code analysis for C/C++ components

### Permissions Required
```yaml
permissions:
  actions: read          # Read workflow artifacts
  contents: read         # Read repository contents
  security-events: write # Write to security events (SARIF upload)
```

### Artifact Retention
CodeQL databases are saved as workflow artifacts:
- **Name**: `codeql-database-{language}`
- **Location**: `${{ runner.temp }}/codeql_databases`
- **Retention**: 7 days

### Configuration Options

#### Excluded Paths
The workflow excludes XML files from analysis:
```yaml
paths-ignore:
  - '**/*.xml'
```

#### Query Suites
Uses the `security-and-quality` query suite for comprehensive analysis:
```yaml
queries: +security-and-quality
```

### Self-Hosted Runner Requirements
Ensure your self-hosted runner has:
1. **Git**: For repository checkout
2. **Java 17+**: For Java project analysis
3. **Build Tools**: Maven or Gradle for Java builds
4. **CodeQL CLI**: Automatically installed by the action
5. **Sufficient Disk Space**: For database creation and artifact storage

### Viewing Results
After workflow completion:
1. Navigate to the **Security** tab in your GitHub repository
2. Select **Code scanning** from the sidebar
3. View detected vulnerabilities and code quality issues
4. Filter by language, severity, or category

### Troubleshooting

#### Database Creation Failures
- Verify JDK installation on self-hosted runner
- Check build tool availability (Maven/Gradle)
- Review build logs for compilation errors

#### Upload Failures
- Verify repository permissions
- Check network connectivity from self-hosted runner
- Ensure SARIF file is generated correctly

#### Timeout Issues
- Adjust `timeout-minutes` if analysis takes longer
- Consider splitting large codebases into multiple workflows
- Optimize build process to reduce analysis time

### Best Practices
1. **Regular Scans**: Keep the weekly schedule enabled
2. **Quick Fixes**: Address high-severity findings promptly
3. **False Positives**: Mark false positives to improve accuracy
4. **Custom Queries**: Add custom queries for project-specific patterns
5. **Runner Maintenance**: Keep self-hosted runners updated

### Additional Resources
- [CodeQL Documentation](https://codeql.github.com/docs/)
- [GitHub Code Scanning](https://docs.github.com/en/code-security/code-scanning)
- [Self-Hosted Runners](https://docs.github.com/en/actions/hosting-your-own-runners)
- [CodeQL CLI Manual](https://codeql.github.com/docs/codeql-cli/)
