🎯 **What:** The vulnerability fixed
Fixed a command injection vulnerability where `subprocess.getoutput` was used with a formatted string containing user-controlled variables.

⚠️ **Risk:** The potential impact if left unfixed
If left unfixed, an attacker who controls the `int8_onnx_path` variable could inject arbitrary shell commands. This could lead to remote code execution (RCE) on the server running the script, resulting in full system compromise, data theft, and loss of integrity.

🛡️ **Solution:** How the fix addresses the vulnerability
Replaced the `subprocess.getoutput(f"curl ...")` calls with `subprocess.run(..., shell=False, check=False)`. By providing the command and arguments as a list of strings instead of a single formatted string, the underlying OS directly executes the command without invoking a shell. This eliminates the possibility of shell metacharacters being interpreted and executed, safely treating the variable value as an argument rather than a command.
