set -euo pipefail

LOCAL_DIR=~/projects/index-tts/
REMOTE_DIR=${LOCAL_DIR}

REMOTE_HOST='Spencer-Desktop'
REMOTE_TARGET=${REMOTE_HOST}:${REMOTE_DIR}

UV_CMD='~/.local/bin/uv'
TEXT="This is a test."
VOICE_FILE='outputs/mizora.ogg'
OUTPUT_FILE='outputs/gen.wav'

rsync --info=NAME -rc -C --exclude='.serena' -f ':- .gitignore' ${LOCAL_DIR} ${REMOTE_TARGET}
ssh -t ${REMOTE_HOST} "time ${UV_CMD} run --directory ${REMOTE_DIR} indextts/cli.py -v ${VOICE_FILE} '${TEXT}' --force -o ${OUTPUT_FILE}"
mkdir -p ${LOCAL_DIR}/outputs
rsync --info=NAME -a ${REMOTE_TARGET}/outputs/gen.wav ${LOCAL_DIR}/outputs/gen.wav
