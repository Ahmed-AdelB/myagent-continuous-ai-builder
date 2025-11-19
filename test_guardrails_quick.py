"""
Quick test to verify guardrails are working before autonomous operation.
"""

from core.orchestrator.guardrails import GuardrailSystem, GuardrailViolation, RiskLevel

def test_guardrails():
    """Test that guardrails block dangerous operations"""

    print("🛡️ Testing Guardrail System...")

    # Initialize in autonomous mode
    guardrails = GuardrailSystem(autonomous_mode=True)

    # Test 1: Should block eval()
    print("\n1. Testing eval() blocking...")
    try:
        result = guardrails.validate_operation(
            operation="data = eval(user_input)",
            operation_type="code"
        )
        if not result['allowed']:
            print("   ✅ PASS: eval() blocked")
        else:
            print("   ❌ FAIL: eval() was allowed!")
            return False
    except GuardrailViolation:
        print("   ✅ PASS: eval() raised GuardrailViolation")

    # Test 2: Should block rm -rf
    print("\n2. Testing rm -rf blocking...")
    try:
        result = guardrails.validate_operation(
            operation="rm -rf /",
            operation_type="file"
        )
        if not result['allowed']:
            print("   ✅ PASS: rm -rf blocked")
        else:
            print("   ❌ FAIL: rm -rf was allowed!")
            return False
    except GuardrailViolation:
        print("   ✅ PASS: rm -rf raised GuardrailViolation")

    # Test 3: Should block force push
    print("\n3. Testing git force push blocking...")
    try:
        result = guardrails.validate_git_operation(
            git_command="git push origin main --force",
            branch="main"
        )
        if not result:
            print("   ✅ PASS: Force push blocked")
        else:
            print("   ❌ FAIL: Force push was allowed!")
            return False
    except GuardrailViolation:
        print("   ✅ PASS: Force push raised GuardrailViolation")

    # Test 4: Should allow safe operations
    print("\n4. Testing safe operations allowed...")
    result = guardrails.validate_operation(
        operation="git status",
        operation_type="git"
    )
    if result['allowed']:
        print("   ✅ PASS: git status allowed")
    else:
        print("   ❌ FAIL: git status was blocked!")
        return False

    # Test 5: Risk assessment
    print("\n5. Testing risk assessment...")
    # Test in non-autonomous mode to check risk level assessment
    non_auto_guardrails = GuardrailSystem(autonomous_mode=False)
    result = non_auto_guardrails.validate_operation(
        operation="DROP DATABASE production",
        operation_type="database"
    )
    if result['risk_level'] == RiskLevel.CRITICAL:
        print("   ✅ PASS: DROP DATABASE assessed as CRITICAL")
    else:
        print(f"   ❌ FAIL: Wrong risk level: {result['risk_level']}")
        return False

    # Test 6: Audit trail
    print("\n6. Testing audit trail...")
    audit = guardrails.get_audit_trail()
    if audit['total_blocked'] >= 3:  # Should have blocked eval, rm -rf, force push
        print(f"   ✅ PASS: Audit trail shows {audit['total_blocked']} blocked operations")
    else:
        print(f"   ❌ FAIL: Expected >= 3 blocked, got {audit['total_blocked']}")
        return False

    print("\n" + "="*60)
    print("🎉 ALL GUARDRAIL TESTS PASSED!")
    print("="*60)
    print(f"\n📊 Audit Summary:")
    print(f"   Blocked operations: {audit['total_blocked']}")
    print(f"   High-risk operations: {audit['total_high_risk']}")
    print(f"   Safe operations executed: {audit['total_executed']}")
    print("\n✅ Guardrails are functioning correctly")
    print("✅ Safe for autonomous operation")

    return True

if __name__ == "__main__":
    import sys
    success = test_guardrails()
    sys.exit(0 if success else 1)
