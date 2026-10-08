# Polls the FluxRT server's /healthz twice a second (Status page, viewer overlay)
# and runs prompt Autoplay.


def onFrameStart(frame):
	if frame % max(1, int(me.time.rate // 2)) == 0:
		op('client').module.poll()
	op('prompts').module.tick()
	return
