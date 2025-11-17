"""
Security & Compliance Scanner - GPT-5 Priority 5
Comprehensive security analysis and compliance monitoring system.

Features:
- Vulnerability detection and classification
- Compliance framework adherence checking
- Code security analysis
- Infrastructure security monitoring
- Security audit automation
- Defensive security recommendations
"""

import asyncio
import json
import hashlib
import re
import subprocess
import threading
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum, auto
import logging

logger = logging.getLogger(__name__)


class ScanType(Enum):
    """Types of security scans that can be performed"""
    CODE_VULNERABILITY = auto()
    DEPENDENCY_CHECK = auto()
    CONFIGURATION_AUDIT = auto()
    COMPLIANCE_CHECK = auto()
    INFRASTRUCTURE_SCAN = auto()
    API_SECURITY = auto()
    DATA_PROTECTION = auto()
    ACCESS_CONTROL = auto()


class VulnerabilityType(Enum):
    """Classification of vulnerability types"""
    SQL_INJECTION = auto()
    XSS = auto()
    CSRF = auto()
    AUTHENTICATION_BYPASS = auto()
    AUTHORIZATION_FLAW = auto()
    INSECURE_CRYPTO = auto()
    HARDCODED_SECRETS = auto()
    INSECURE_DESERIALIZATION = auto()
    UNSAFE_REFLECTION = auto()
    PATH_TRAVERSAL = auto()
    COMMAND_INJECTION = auto()
    LDAP_INJECTION = auto()
    XML_INJECTION = auto()
    BUFFER_OVERFLOW = auto()
    RACE_CONDITION = auto()
    INSECURE_RANDOM = auto()
    WEAK_HASH = auto()
    INSECURE_TRANSPORT = auto()
    INFORMATION_DISCLOSURE = auto()
    DENIAL_OF_SERVICE = auto()


class SecurityLevel(Enum):
    """Security vulnerability severity levels"""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"
    INFO = "INFO"


class ComplianceFramework(Enum):
    """Supported compliance frameworks"""
    GDPR = "GDPR"
    HIPAA = "HIPAA"
    PCI_DSS = "PCI-DSS"
    SOX = "SOX"
    ISO_27001 = "ISO-27001"
    NIST = "NIST"
    OWASP = "OWASP"
    CIS = "CIS"
    SOC_2 = "SOC-2"
    FISMA = "FISMA"


@dataclass
class SecurityVulnerability:
    """Represents a detected security vulnerability"""
    id: str
    type: VulnerabilityType
    severity: SecurityLevel
    title: str
    description: str
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    code_snippet: Optional[str] = None
    remediation: str = ""
    cvss_score: Optional[float] = None
    cve_id: Optional[str] = None
    references: List[str] = field(default_factory=list)
    detected_at: datetime = field(default_factory=datetime.utcnow)
    false_positive: bool = False


@dataclass
class ComplianceRule:
    """Represents a compliance rule to be checked"""
    id: str
    framework: ComplianceFramework
    title: str
    description: str
    requirement: str
    check_function: str
    severity: SecurityLevel = SecurityLevel.MEDIUM
    automated: bool = True
    tags: List[str] = field(default_factory=list)


@dataclass
class SecurityScanResult:
    """Results from a security scan"""
    scan_id: str
    scan_type: ScanType
    target: str
    start_time: datetime
    end_time: datetime
    vulnerabilities: List[SecurityVulnerability] = field(default_factory=list)
    total_files_scanned: int = 0
    scan_duration_seconds: float = 0.0
    status: str = "completed"
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ComplianceCheckResult:
    """Results from a compliance check"""
    check_id: str
    rule: ComplianceRule
    status: str  # "pass", "fail", "warning", "not_applicable"
    findings: List[str] = field(default_factory=list)
    evidence: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)
    checked_at: datetime = field(default_factory=datetime.utcnow)


class SecurityComplianceScanner:
    """
    Comprehensive security and compliance scanner for defensive security analysis.

    Capabilities:
    - Static code analysis for vulnerabilities
    - Dependency vulnerability scanning
    - Configuration security assessment
    - Compliance framework checking
    - Infrastructure security monitoring
    - API security analysis
    """

    def __init__(self, config_path: Optional[str] = None, telemetry=None):
        self.config_path = config_path
        self.telemetry = telemetry
        self.scan_history: List[SecurityScanResult] = []
        self.compliance_results: List[ComplianceCheckResult] = []
        self.vulnerability_patterns = self._load_vulnerability_patterns()
        self.compliance_rules = self._load_compliance_rules()
        self.scan_lock = threading.Lock()

        # Security scan metrics
        self.metrics = {
            'total_scans': 0,
            'vulnerabilities_found': 0,
            'compliance_checks_passed': 0,
            'compliance_checks_failed': 0,
            'scan_duration_avg': 0.0
        }

        logger.info("Security & Compliance Scanner initialized")

    def _load_vulnerability_patterns(self) -> Dict[VulnerabilityType, List[Dict]]:
        """Load vulnerability detection patterns for defensive security analysis"""
        return {
            VulnerabilityType.SQL_INJECTION: [
                {
                    'pattern': r'(?i)(select|insert|update|delete|union|drop)\s+.*\+.*\$',
                    'description': 'Potential SQL injection via string concatenation',
                    'severity': SecurityLevel.HIGH,
                    'cwe': 'CWE-89'
                },
                {
                    'pattern': r'(?i)execute\s*\(\s*["\'].*\+.*["\']',
                    'description': 'SQL execution with concatenated user input',
                    'severity': SecurityLevel.CRITICAL,
                    'cwe': 'CWE-89'
                }
            ],
            VulnerabilityType.XSS: [
                {
                    'pattern': r'innerHTML\s*=\s*.*\+|document\.write\s*\(.*\+',
                    'description': 'Potential DOM-based XSS vulnerability',
                    'severity': SecurityLevel.HIGH,
                    'cwe': 'CWE-79'
                },
                {
                    'pattern': r'dangerouslySetInnerHTML\s*:\s*{\s*__html:',
                    'description': 'React dangerouslySetInnerHTML usage',
                    'severity': SecurityLevel.MEDIUM,
                    'cwe': 'CWE-79'
                }
            ],
            VulnerabilityType.HARDCODED_SECRETS: [
                {
                    'pattern': r'(?i)(password|pwd|secret|key|token)\s*=\s*["\'][^"\']{8,}["\']',
                    'description': 'Hardcoded credential detected',
                    'severity': SecurityLevel.CRITICAL,
                    'cwe': 'CWE-798'
                },
                {
                    'pattern': r'(?i)api[_-]?key\s*=\s*["\'][a-zA-Z0-9]{20,}["\']',
                    'description': 'Hardcoded API key detected',
                    'severity': SecurityLevel.HIGH,
                    'cwe': 'CWE-798'
                }
            ],
            VulnerabilityType.INSECURE_CRYPTO: [
                {
                    'pattern': r'(?i)(md5|sha1|des|rc4)\s*\(',
                    'description': 'Weak cryptographic algorithm usage',
                    'severity': SecurityLevel.MEDIUM,
                    'cwe': 'CWE-327'
                },
                {
                    'pattern': r'(?i)random\(\)\s*\*|Math\.random\(\)',
                    'description': 'Insecure random number generation',
                    'severity': SecurityLevel.MEDIUM,
                    'cwe': 'CWE-338'
                }
            ],
            VulnerabilityType.COMMAND_INJECTION: [
                {
                    'pattern': r'(?i)(exec|system|shell_exec|eval)\s*\(.*\$',
                    'description': 'Command injection via user input',
                    'severity': SecurityLevel.CRITICAL,
                    'cwe': 'CWE-78'
                },
                {
                    'pattern': r'subprocess\.(call|run|Popen).*shell\s*=\s*True',
                    'description': 'Shell command execution with shell=True',
                    'severity': SecurityLevel.HIGH,
                    'cwe': 'CWE-78'
                }
            ],
            VulnerabilityType.PATH_TRAVERSAL: [
                {
                    'pattern': r'open\s*\(.*\+|file\s*\(.*\+',
                    'description': 'File operation with concatenated path',
                    'severity': SecurityLevel.HIGH,
                    'cwe': 'CWE-22'
                },
                {
                    'pattern': r'\.\.\/|\.\.\\',
                    'description': 'Directory traversal patterns detected',
                    'severity': SecurityLevel.MEDIUM,
                    'cwe': 'CWE-22'
                }
            ],
            VulnerabilityType.INSECURE_TRANSPORT: [
                {
                    'pattern': r'(?i)http:\/\/|ssl\s*=\s*false|verify\s*=\s*false',
                    'description': 'Insecure transport configuration',
                    'severity': SecurityLevel.MEDIUM,
                    'cwe': 'CWE-319'
                }
            ]
        }

    def _load_compliance_rules(self) -> Dict[ComplianceFramework, List[ComplianceRule]]:
        """Load compliance rules for various frameworks"""
        rules = {
            ComplianceFramework.OWASP: [
                ComplianceRule(
                    id="OWASP-A01-2021",
                    framework=ComplianceFramework.OWASP,
                    title="Broken Access Control",
                    description="Access control enforces policy such that users cannot act outside of their intended permissions",
                    requirement="Implement proper access controls and authorization checks",
                    check_function="check_access_control",
                    severity=SecurityLevel.HIGH,
                    tags=["access-control", "authorization"]
                ),
                ComplianceRule(
                    id="OWASP-A02-2021",
                    framework=ComplianceFramework.OWASP,
                    title="Cryptographic Failures",
                    description="Protect sensitive data through proper encryption",
                    requirement="Use strong encryption for data at rest and in transit",
                    check_function="check_cryptographic_implementations",
                    severity=SecurityLevel.HIGH,
                    tags=["encryption", "crypto"]
                ),
                ComplianceRule(
                    id="OWASP-A03-2021",
                    framework=ComplianceFramework.OWASP,
                    title="Injection",
                    description="Application is vulnerable to injection attacks",
                    requirement="Validate and sanitize all user inputs",
                    check_function="check_injection_vulnerabilities",
                    severity=SecurityLevel.CRITICAL,
                    tags=["injection", "input-validation"]
                )
            ],
            ComplianceFramework.GDPR: [
                ComplianceRule(
                    id="GDPR-Art6",
                    framework=ComplianceFramework.GDPR,
                    title="Lawfulness of processing",
                    description="Personal data processing must have legal basis",
                    requirement="Implement lawful basis tracking for personal data processing",
                    check_function="check_data_processing_basis",
                    severity=SecurityLevel.HIGH,
                    tags=["privacy", "data-processing"]
                ),
                ComplianceRule(
                    id="GDPR-Art32",
                    framework=ComplianceFramework.GDPR,
                    title="Security of processing",
                    description="Implement appropriate technical and organizational measures",
                    requirement="Encrypt personal data and implement access controls",
                    check_function="check_data_security_measures",
                    severity=SecurityLevel.HIGH,
                    tags=["security", "encryption", "access-control"]
                )
            ],
            ComplianceFramework.PCI_DSS: [
                ComplianceRule(
                    id="PCI-DSS-3.4",
                    framework=ComplianceFramework.PCI_DSS,
                    title="Protect cardholder data",
                    description="Render PAN unreadable anywhere it is stored",
                    requirement="Encrypt cardholder data using strong cryptography",
                    check_function="check_cardholder_data_protection",
                    severity=SecurityLevel.CRITICAL,
                    tags=["pci", "encryption", "cardholder-data"]
                )
            ]
        }
        return rules

    async def scan_code_vulnerabilities(self, target_path: str, scan_types: Optional[List[ScanType]] = None) -> SecurityScanResult:
        """
        Perform comprehensive code vulnerability scanning for defensive security analysis.

        Args:
            target_path: Path to scan (file or directory)
            scan_types: Specific scan types to perform

        Returns:
            SecurityScanResult with detected vulnerabilities
        """
        scan_id = hashlib.md5(f"{target_path}_{datetime.utcnow()}".encode()).hexdigest()
        start_time = datetime.utcnow()

        logger.info(f"Starting security scan: {scan_id} on {target_path}")

        if self.telemetry:
            self.telemetry.record_event("security_scan_started", {
                'scan_id': scan_id,
                'target': target_path,
                'scan_types': [st.name for st in (scan_types or [])]
            })

        vulnerabilities = []
        files_scanned = 0

        try:
            with self.scan_lock:
                if Path(target_path).is_file():
                    vulns = await self._scan_file(target_path)
                    vulnerabilities.extend(vulns)
                    files_scanned = 1
                elif Path(target_path).is_dir():
                    for file_path in Path(target_path).rglob('*'):
                        if file_path.is_file() and self._should_scan_file(file_path):
                            vulns = await self._scan_file(str(file_path))
                            vulnerabilities.extend(vulns)
                            files_scanned += 1

                            # Yield control to prevent blocking
                            if files_scanned % 50 == 0:
                                await asyncio.sleep(0)
                else:
                    logger.warning(f"Invalid scan target: {target_path}")

            end_time = datetime.utcnow()
            duration = (end_time - start_time).total_seconds()

            # Update metrics
            self.metrics['total_scans'] += 1
            self.metrics['vulnerabilities_found'] += len(vulnerabilities)
            self.metrics['scan_duration_avg'] = (
                (self.metrics['scan_duration_avg'] * (self.metrics['total_scans'] - 1) + duration)
                / self.metrics['total_scans']
            )

            scan_result = SecurityScanResult(
                scan_id=scan_id,
                scan_type=ScanType.CODE_VULNERABILITY,
                target=target_path,
                start_time=start_time,
                end_time=end_time,
                vulnerabilities=vulnerabilities,
                total_files_scanned=files_scanned,
                scan_duration_seconds=duration,
                metadata={
                    'vulnerability_types': list(set(v.type.name for v in vulnerabilities)),
                    'severity_breakdown': self._calculate_severity_breakdown(vulnerabilities)
                }
            )

            self.scan_history.append(scan_result)

            if self.telemetry:
                self.telemetry.record_event("security_scan_completed", {
                    'scan_id': scan_id,
                    'vulnerabilities_found': len(vulnerabilities),
                    'files_scanned': files_scanned,
                    'duration_seconds': duration
                })

            logger.info(f"Security scan completed: {len(vulnerabilities)} vulnerabilities found in {files_scanned} files")

            return scan_result

        except Exception as e:
            logger.error(f"Security scan failed: {e}")
            if self.telemetry:
                self.telemetry.record_event("security_scan_failed", {
                    'scan_id': scan_id,
                    'error': str(e)
                })
            raise

    async def _scan_file(self, file_path: str) -> List[SecurityVulnerability]:
        """Scan a single file for vulnerabilities"""
        vulnerabilities = []

        try:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()

            lines = content.split('\n')

            # Check each vulnerability pattern
            for vuln_type, patterns in self.vulnerability_patterns.items():
                for pattern_info in patterns:
                    pattern = pattern_info['pattern']

                    for line_num, line in enumerate(lines, 1):
                        matches = re.finditer(pattern, line, re.IGNORECASE | re.MULTILINE)

                        for match in matches:
                            vuln = SecurityVulnerability(
                                id=f"{vuln_type.name}_{file_path}_{line_num}_{match.start()}",
                                type=vuln_type,
                                severity=pattern_info['severity'],
                                title=f"{vuln_type.name.replace('_', ' ').title()} Detected",
                                description=pattern_info['description'],
                                file_path=file_path,
                                line_number=line_num,
                                code_snippet=line.strip(),
                                remediation=self._get_remediation_advice(vuln_type),
                                references=[
                                    f"CWE: {pattern_info.get('cwe', 'N/A')}",
                                    f"Pattern: {pattern}"
                                ]
                            )
                            vulnerabilities.append(vuln)

        except Exception as e:
            logger.warning(f"Failed to scan file {file_path}: {e}")

        return vulnerabilities

    def _should_scan_file(self, file_path: Path) -> bool:
        """Determine if a file should be scanned based on extension and content"""
        # Scan common code file types
        scan_extensions = {
            '.py', '.js', '.ts', '.jsx', '.tsx', '.java', '.cs', '.cpp', '.c',
            '.php', '.rb', '.go', '.rs', '.swift', '.kt', '.scala', '.sql',
            '.sh', '.bash', '.ps1', '.yaml', '.yml', '.json', '.xml', '.html'
        }

        # Skip binary and large files
        skip_extensions = {
            '.exe', '.dll', '.so', '.dylib', '.jar', '.war', '.ear',
            '.pdf', '.doc', '.docx', '.xls', '.xlsx', '.ppt', '.pptx',
            '.jpg', '.jpeg', '.png', '.gif', '.bmp', '.svg', '.ico',
            '.mp3', '.mp4', '.avi', '.mov', '.wmv', '.zip', '.rar', '.tar', '.gz'
        }

        if file_path.suffix.lower() in skip_extensions:
            return False

        if file_path.suffix.lower() in scan_extensions:
            return True

        # Check if file appears to be text-based
        try:
            if file_path.stat().st_size > 10 * 1024 * 1024:  # Skip files > 10MB
                return False

            with open(file_path, 'rb') as f:
                sample = f.read(1024)
                if b'\0' in sample:  # Likely binary file
                    return False

            return True
        except:
            return False

    async def check_compliance(self,
                              frameworks: List[ComplianceFramework],
                              target_path: str) -> List[ComplianceCheckResult]:
        """
        Perform compliance checking against specified frameworks.

        Args:
            frameworks: List of compliance frameworks to check
            target_path: Path to analyze for compliance

        Returns:
            List of compliance check results
        """
        results = []

        logger.info(f"Starting compliance check for frameworks: {[f.value for f in frameworks]}")

        for framework in frameworks:
            if framework in self.compliance_rules:
                for rule in self.compliance_rules[framework]:
                    try:
                        result = await self._execute_compliance_check(rule, target_path)
                        results.append(result)

                        # Update metrics
                        if result.status == "pass":
                            self.metrics['compliance_checks_passed'] += 1
                        elif result.status == "fail":
                            self.metrics['compliance_checks_failed'] += 1

                    except Exception as e:
                        logger.error(f"Compliance check failed for {rule.id}: {e}")

                        result = ComplianceCheckResult(
                            check_id=rule.id,
                            rule=rule,
                            status="error",
                            findings=[f"Check execution failed: {e}"],
                            recommendations=["Review compliance check implementation"]
                        )
                        results.append(result)

        self.compliance_results.extend(results)

        logger.info(f"Compliance check completed: {len(results)} rules checked")

        return results

    async def _execute_compliance_check(self, rule: ComplianceRule, target_path: str) -> ComplianceCheckResult:
        """Execute a single compliance check"""
        # This is a simplified implementation - real compliance checking would be more sophisticated

        if rule.check_function == "check_injection_vulnerabilities":
            # Check for injection vulnerabilities using our scanner
            scan_result = await self.scan_code_vulnerabilities(target_path)
            injection_vulns = [v for v in scan_result.vulnerabilities
                             if v.type in [VulnerabilityType.SQL_INJECTION,
                                         VulnerabilityType.COMMAND_INJECTION,
                                         VulnerabilityType.LDAP_INJECTION,
                                         VulnerabilityType.XML_INJECTION]]

            if injection_vulns:
                return ComplianceCheckResult(
                    check_id=rule.id,
                    rule=rule,
                    status="fail",
                    findings=[f"Found {len(injection_vulns)} injection vulnerabilities"],
                    evidence=[f"{v.file_path}:{v.line_number}" for v in injection_vulns],
                    recommendations=[
                        "Implement input validation and parameterized queries",
                        "Use prepared statements for database access",
                        "Sanitize user inputs before processing"
                    ]
                )
            else:
                return ComplianceCheckResult(
                    check_id=rule.id,
                    rule=rule,
                    status="pass",
                    findings=["No injection vulnerabilities detected"]
                )

        elif rule.check_function == "check_cryptographic_implementations":
            # Check for weak crypto
            scan_result = await self.scan_code_vulnerabilities(target_path)
            crypto_vulns = [v for v in scan_result.vulnerabilities
                          if v.type == VulnerabilityType.INSECURE_CRYPTO]

            if crypto_vulns:
                return ComplianceCheckResult(
                    check_id=rule.id,
                    rule=rule,
                    status="fail",
                    findings=[f"Found {len(crypto_vulns)} cryptographic weaknesses"],
                    evidence=[f"{v.file_path}:{v.line_number}" for v in crypto_vulns],
                    recommendations=[
                        "Use strong cryptographic algorithms (AES-256, SHA-256+)",
                        "Implement proper key management",
                        "Use cryptographically secure random number generators"
                    ]
                )
            else:
                return ComplianceCheckResult(
                    check_id=rule.id,
                    rule=rule,
                    status="pass",
                    findings=["No weak cryptographic implementations detected"]
                )

        # Default implementation for other checks
        return ComplianceCheckResult(
            check_id=rule.id,
            rule=rule,
            status="not_applicable",
            findings=["Automated check not implemented"],
            recommendations=["Manual review required"]
        )

    def _get_remediation_advice(self, vuln_type: VulnerabilityType) -> str:
        """Get remediation advice for a vulnerability type"""
        remediation_map = {
            VulnerabilityType.SQL_INJECTION: "Use parameterized queries and input validation",
            VulnerabilityType.XSS: "Sanitize user inputs and use content security policy",
            VulnerabilityType.HARDCODED_SECRETS: "Move secrets to environment variables or secure storage",
            VulnerabilityType.INSECURE_CRYPTO: "Use strong cryptographic algorithms and proper implementation",
            VulnerabilityType.COMMAND_INJECTION: "Validate inputs and avoid shell execution",
            VulnerabilityType.PATH_TRAVERSAL: "Validate file paths and use allowlists",
            VulnerabilityType.INSECURE_TRANSPORT: "Use HTTPS/TLS for all communications"
        }
        return remediation_map.get(vuln_type, "Follow security best practices")

    def _calculate_severity_breakdown(self, vulnerabilities: List[SecurityVulnerability]) -> Dict[str, int]:
        """Calculate breakdown of vulnerabilities by severity"""
        breakdown = {level.value: 0 for level in SecurityLevel}
        for vuln in vulnerabilities:
            breakdown[vuln.severity.value] += 1
        return breakdown

    async def generate_security_report(self,
                                     scan_results: List[SecurityScanResult],
                                     compliance_results: List[ComplianceCheckResult]) -> Dict[str, Any]:
        """Generate comprehensive security and compliance report"""

        all_vulnerabilities = []
        for result in scan_results:
            all_vulnerabilities.extend(result.vulnerabilities)

        severity_breakdown = self._calculate_severity_breakdown(all_vulnerabilities)

        # Compliance summary
        compliance_summary = {
            'total_checks': len(compliance_results),
            'passed': len([r for r in compliance_results if r.status == 'pass']),
            'failed': len([r for r in compliance_results if r.status == 'fail']),
            'warnings': len([r for r in compliance_results if r.status == 'warning']),
            'not_applicable': len([r for r in compliance_results if r.status == 'not_applicable'])
        }

        report = {
            'report_id': hashlib.md5(f"security_report_{datetime.utcnow()}".encode()).hexdigest(),
            'generated_at': datetime.utcnow().isoformat(),
            'summary': {
                'total_vulnerabilities': len(all_vulnerabilities),
                'critical_vulnerabilities': severity_breakdown['CRITICAL'],
                'high_vulnerabilities': severity_breakdown['HIGH'],
                'medium_vulnerabilities': severity_breakdown['MEDIUM'],
                'low_vulnerabilities': severity_breakdown['LOW'],
                'info_findings': severity_breakdown['INFO'],
                'security_score': self._calculate_security_score(all_vulnerabilities),
                'compliance_score': self._calculate_compliance_score(compliance_results)
            },
            'vulnerability_breakdown': severity_breakdown,
            'compliance_summary': compliance_summary,
            'top_vulnerabilities': sorted(all_vulnerabilities,
                                        key=lambda v: (v.severity.name, v.detected_at),
                                        reverse=True)[:10],
            'failed_compliance_checks': [r for r in compliance_results if r.status == 'fail'],
            'recommendations': self._generate_recommendations(all_vulnerabilities, compliance_results),
            'scan_metadata': {
                'total_scans': len(scan_results),
                'total_files_scanned': sum(r.total_files_scanned for r in scan_results),
                'total_scan_time': sum(r.scan_duration_seconds for r in scan_results)
            }
        }

        return report

    def _calculate_security_score(self, vulnerabilities: List[SecurityVulnerability]) -> float:
        """Calculate overall security score (0-100)"""
        if not vulnerabilities:
            return 100.0

        # Weight vulnerabilities by severity
        weights = {
            SecurityLevel.CRITICAL: 10,
            SecurityLevel.HIGH: 7,
            SecurityLevel.MEDIUM: 4,
            SecurityLevel.LOW: 2,
            SecurityLevel.INFO: 1
        }

        total_weight = sum(weights[v.severity] for v in vulnerabilities)
        max_possible = len(vulnerabilities) * weights[SecurityLevel.CRITICAL]

        if max_possible == 0:
            return 100.0

        score = max(0, 100 - (total_weight / max_possible * 100))
        return round(score, 2)

    def _calculate_compliance_score(self, results: List[ComplianceCheckResult]) -> float:
        """Calculate compliance score (0-100)"""
        if not results:
            return 100.0

        passed = len([r for r in results if r.status == 'pass'])
        total_applicable = len([r for r in results if r.status != 'not_applicable'])

        if total_applicable == 0:
            return 100.0

        return round((passed / total_applicable) * 100, 2)

    def _generate_recommendations(self,
                                vulnerabilities: List[SecurityVulnerability],
                                compliance_results: List[ComplianceCheckResult]) -> List[str]:
        """Generate security and compliance recommendations"""
        recommendations = []

        # Vulnerability-based recommendations
        if any(v.severity == SecurityLevel.CRITICAL for v in vulnerabilities):
            recommendations.append("🚨 Address all CRITICAL vulnerabilities immediately")

        if any(v.type == VulnerabilityType.HARDCODED_SECRETS for v in vulnerabilities):
            recommendations.append("🔐 Implement secure secret management system")

        if any(v.type in [VulnerabilityType.SQL_INJECTION, VulnerabilityType.XSS] for v in vulnerabilities):
            recommendations.append("🛡️ Implement comprehensive input validation and sanitization")

        # Compliance-based recommendations
        failed_checks = [r for r in compliance_results if r.status == 'fail']
        if failed_checks:
            recommendations.append(f"📋 Address {len(failed_checks)} failed compliance checks")

        if len(vulnerabilities) > 50:
            recommendations.append("🔄 Implement automated security testing in CI/CD pipeline")

        if not recommendations:
            recommendations.append("✅ Maintain current security posture with regular scanning")

        return recommendations

    def get_scan_metrics(self) -> Dict[str, Any]:
        """Get security scanning metrics"""
        return {
            **self.metrics,
            'scan_history_count': len(self.scan_history),
            'compliance_results_count': len(self.compliance_results),
            'last_scan_time': self.scan_history[-1].end_time.isoformat() if self.scan_history else None
        }

    async def export_results(self, format_type: str = "json") -> str:
        """Export scan results and compliance data"""
        export_data = {
            'scan_history': [
                {
                    'scan_id': result.scan_id,
                    'scan_type': result.scan_type.name,
                    'target': result.target,
                    'vulnerabilities_count': len(result.vulnerabilities),
                    'start_time': result.start_time.isoformat(),
                    'end_time': result.end_time.isoformat(),
                    'duration_seconds': result.scan_duration_seconds
                } for result in self.scan_history
            ],
            'compliance_results': [
                {
                    'check_id': result.check_id,
                    'framework': result.rule.framework.value,
                    'title': result.rule.title,
                    'status': result.status,
                    'findings_count': len(result.findings),
                    'checked_at': result.checked_at.isoformat()
                } for result in self.compliance_results
            ],
            'metrics': self.get_scan_metrics()
        }

        if format_type == "json":
            return json.dumps(export_data, indent=2)
        else:
            raise ValueError(f"Unsupported format: {format_type}")