REALIZE_ASAP = True

def maybe_realize(t):
	if REALIZE_ASAP:
		return t.realize()
