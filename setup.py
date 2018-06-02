from setuptools import setup
from setuptools.command.install import install


class ScriptInstall(install):
    def run(self):
        install.run(self)
        print("\n\n\n\nCustom Installer\n\n\n\n")


setup(
    name='eunomia',
    version='0.1',
    packages=['eunomia', 'eunomia.sim_tools'],
    url='https://github.com/pierfied/eunomia',
    license='',
    author='Pier Fiedorowicz',
    author_email='pierfied@email.arizona.edu',
    description='',
    cmdclass={'install': ScriptInstall}
)
