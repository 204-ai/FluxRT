# webclient_prompts replies (GET /prompts).


def onConnect(webClientDAT):
	return


def onDisconnect(webClientDAT):
	return


def onResponse(webClientDAT, statusCode, headerDict, data):
	op('prompts').module.on_prompts(statusCode, data)
	return
