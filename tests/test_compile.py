import subprocess, tempfile, os
def test_cpp_compiles():
    code = "#include <iostream>\nint main(){std::cout<<\"ok\";}\n"
    with tempfile.TemporaryDirectory() as d:
        cpp=os.path.join(d,"a.cpp"); exe=os.path.join(d,"a.out")
        open(cpp,"w").write(code)
        r = subprocess.run(f"g++ {cpp} -o {exe}", shell=True)
        assert r.returncode==0
