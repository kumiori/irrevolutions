# %%capture
import sys
import subprocess

try:
    import google.colab  # noqa: F401
except ImportError:
    import ufl  # noqa: F401
    import dolfinx  # noqa: F401
else:
    try:
        import ufl
        import dolfinx
    except ImportError:
        # !wget "https://fem-on-colab.github.io/releases/fenicsx-install.sh" -O "/tmp/fenicsx-install.sh" && bash "/tmp/fenicsx-install.sh";
        ret = subprocess.run(["wget", "https://fem-on-colab.github.io/releases/fenicsx-install.sh", "-O /tmp/fenicsx-install.sh"])
        ret = subprocess.run(["bash", "/tmp/fenicsx-install.sh"])
        import ufl  # noqa: F401
        import dolfinx  # noqa: F401

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