"""Run all BiBo tests"""
import subprocess
import sys

tests = [
    ("Modular components", "tests/test_modular.py"),
    ("RoPE implementation", "tests/test_rope.py"),
    ("SSMax math", "tests/ssmax/test_ssmax_math_only.py"),
    ("SSMax vs standard", "tests/ssmax/test_ssmax_vs_standard.py"),
    ("Attention output", "tests/ssmax/test_attention_output.py"),
    ("Comprehensive forward", "tests/ssmax/test_comprehensive_forward.py"),
]

print("=" * 60)
print("Running BiBo Test Suite")
print("=" * 60)

results = []
for name, path in tests:
    print(f"\n▶ {name}...")
    result = subprocess.run(
        [sys.executable, path],
        capture_output=True,
        text=True
    )
    
    # Check for pass indicators
    output = result.stdout + result.stderr
    passed = "✅" in output or "All tests passed" in output or "passed" in output.lower()
    
    if passed and result.returncode == 0:
        print(f"  ✅ PASS")
        results.append((name, True))
    else:
        print(f"  ❌ FAIL")
        # Show error
        if "Error" in output or "Traceback" in output:
            lines = output.split('\n')
            for i, line in enumerate(lines):
                if "Error" in line or "Traceback" in line:
                    print(f"     {line}")
                    if i + 1 < len(lines):
                        print(f"     {lines[i+1]}")
                    break
        results.append((name, False))

print("\n" + "=" * 60)
print("Test Summary")
print("=" * 60)

passed = sum(1 for _, p in results if p)
total = len(results)

for name, p in results:
    status = "✅" if p else "❌"
    print(f"{status} {name}")

print(f"\n{passed}/{total} test suites passed")

if passed == total:
    print("\n🎉 All tests passed!")
    sys.exit(0)
else:
    print(f"\n⚠️  {total - passed} test suite(s) failed")
    sys.exit(1)
