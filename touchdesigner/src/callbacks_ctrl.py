# webclient_ctrl replies (POST /prompt, /prompt-travel, /seed, /steps).


def onConnect(webClientDAT):
	return


def onDisconnect(webClientDAT):
	return


def onResponse(webClientDAT, statusCode, headerDict, data):
	op('client').module.on_ctrl(statusCode, data)
	return
