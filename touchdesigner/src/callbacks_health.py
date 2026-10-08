# webclient_health replies (GET /healthz).


def onConnect(webClientDAT):
	return


def onDisconnect(webClientDAT):
	return


def onResponse(webClientDAT, statusCode, headerDict, data):
	op('client').module.on_health(statusCode, data)
	return
