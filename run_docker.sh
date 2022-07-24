docker build -t sspc .
nvidia-docker run \
	--rm \
	-v ${PWD}/data:/data \
	-v ${PWD}/codes:/codes \
	-it sspc;
