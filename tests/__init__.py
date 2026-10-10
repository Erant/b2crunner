import os

# The image sets these (docker/Dockerfile) and they win over B2C_DATA_DIR, so
# a test that points B2C_DATA_DIR at a temp volume would still read and write
# the container's /data — archives, readiness markers and all. Every path
# derives from B2C_DATA_DIR once they are gone.
for _name in ("B2C_OUTPUT_DIR", "B2C_LOG_DIR", "B2C_UPLOAD_DIR", "B2C_MODELS_DIR",
              "B2C_RUN_JOBS_DIR"):
    os.environ.pop(_name, None)
