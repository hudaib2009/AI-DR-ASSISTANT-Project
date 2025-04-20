import tensorflow as tf # type: ignore

# Check CPU availability
cpus = tf.config.list_physical_devices('CPU')
if cpus:
    print("CPUs Available: ", len(cpus))
    print("CPU Details: ", cpus)
else:
    print("No CPUs Available.")

# Check GPU availability
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs Available: ", len(gpus))
    print("GPU Details: ", gpus)
else:
    print("No GPUs Available.")