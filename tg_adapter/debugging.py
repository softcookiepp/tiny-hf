REALIZE_ASAP = True
KEEP_INPUT_TENSORS = True

def maybe_realize(t):
	if REALIZE_ASAP:
		if hasattr(t, "realize"):
			return t.realize()
		elif hasattr(t, "tg"):
			t.tg.realize()
			return t
		raise ValueError
	return t

