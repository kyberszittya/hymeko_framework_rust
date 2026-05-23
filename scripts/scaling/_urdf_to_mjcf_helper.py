
import sys, mujoco
spec = mujoco.MjSpec.from_file(sys.argv[1])
sys.stdout.write(spec.to_xml())
