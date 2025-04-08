import tinygrad

def recursive_realize(self, *args):
	input(args)
	new_args = []
	for arg in args:
		if isinstance(arg, tinygrad.Tensor):
			new_args.append(arg.realize() )
		elif isinstance(arg, tuple):
			new_args.append(recursive_realize(*arg) )
		elif isinstance(arg, list):
			new_args.append(list(recursive_realize(*tuple(arg) ) ) )
		elif isinstance(arg, dict):
			new_arg = {}
			for k, v in arg.items():
				new_arg[k] = recursive_realize(v)
			new_args.append(new_arg)
		elif hasattr(arg, "__dict__"):
			arg.__dict__.update(recursive_realize(arg.__dict__) )
			new_args.append(arg)
		else:
			# just append as is
			new_args.append(arg)
	assert len(new_args) == len(args)
	if len(new_args) > 1:
		return tuple(new_args)
	return new_args[0]
