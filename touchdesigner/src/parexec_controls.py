# Custom parameter changes on the fluxrt COMP -> FluxRT server (client DAT) and
# the prompt list (prompts DAT).


def onValueChange(par, prev):
	client = op('client').module
	prompts = op('prompts').module
	if par.name in ('Filter', 'Search'):
		prompts.refresh_menu()
		return
	if par.name == 'Autoplay':
		prompts.reset_timer()
		return
	if client.quiet():
		return
	if par.name == 'Prompt':
		client.send_prompt()
	elif par.name == 'Promptindex':
		if not prompts.quiet():
			prompts.apply_index(int(par.eval()))
	elif par.name == 'Seed':
		client.send_seed()
	elif par.name == 'Steps':
		client.send_steps()
	elif par.name == 'Server':
		client.poll()
	return


def onPulse(par):
	client = op('client').module
	prompts = op('prompts').module
	if par.name == 'Send':
		client.send_prompt()
	elif par.name == 'Pullstate':
		client.pull_state()
	elif par.name == 'Loadfile':
		prompts.load_file()
	elif par.name == 'Loadserver':
		prompts.load_server()
	elif par.name == 'Prev':
		prompts.step(-1)
	elif par.name == 'Next':
		prompts.step(1)
	elif par.name == 'Shuffle':
		prompts.shuffle()
	return
