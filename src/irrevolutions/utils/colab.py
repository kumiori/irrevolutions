# %%capture
import subprocess
import sys

try:
    import google.colab  # noqa: F401
except ImportError:
    import dolfinx  # noqa: F401
    import ufl  # noqa: F401
else:
    try:
        import dolfinx
        import ufl
    except ImportError:
        # !wget "https://fem-on-colab.github.io/releases/fenicsx-install.sh" -O "/tmp/fenicsx-install.sh" && bash "/tmp/fenicsx-install.sh";
        try:
            subprocess.run(["wget", "https://fem-on-colab.github.io/releases/fenicsx-install.sh", "-O /tmp/fenicsx-install.sh"])
        except subprocess.CalledProcessError as ret:
            print("error code", ret.returncode, ret.output)

        try:
            subprocess.run(["bash", "/tmp/fenicsx-install.sh"])
        except subprocess.CalledProcessError as ret:
            print("error code", ret.returncode, ret.output)

        import dolfinx  # noqa: F401
        import ufl  # noqa: F401

# !sudo apt install libgl1-mesa-glx xvfb;
# !{sys.executable} -m pip install pythreejs;
# !{sys.executable} -m pip install ipygany;

# try:
#     import google.colab
# except ImportError:
#     pass
# else:
#     google.colab.output.enable_custom_widget_manager();
# try:
#     import pyvista
# except ImportError:
#     !pip3 install --upgrade pyvista itkwidgets;
#     import pyvista  # noqa: F401