from .module import Module
from typing import Dict, Optional, Iterable, Union, Iterator
from overloading import overload
import tinygrad

# Just do this for now, since Tinygrad's state dict loader doesn't
# seem to like it when we use subclasses of list
ModuleList = list

"""
class ModuleList(list, Module):
	def __init__(self, *args):
		super().__init__(*args)
		print("me stuff")
		for item in self:
			print(item)
		print(self.parameters() )
		input("oof")
"""
'''
class ModuleList(Module):
	r"""Holds submodules in a list.

	:class:`~torch.nn.ModuleList` can be indexed like a regular Python list, but
	modules it contains are properly registered, and will be visible by all
	:class:`~torch.nn.Module` methods.

	Args:
		modules (iterable, optional): an iterable of modules to add

	Example::

		class MyModule(nn.Module):
			def __init__(self) -> None:
				super().__init__()
				self.linears = nn.ModuleList([nn.Linear(10, 10) for i in range(10)])

			def forward(self, x):
				# ModuleList can act as an iterable, or be indexed using ints
				for i, l in enumerate(self.linears):
					x = self.linears[i // 2](x) + l(x)
				return x
	"""

	#_modules: Dict[str, Module]  # type: ignore[assignment]

	def __init__(self, modules: Optional[Iterable[Module]] = None):
		super().__init__()
		self._modules = list(modules)

	def _get_abs_string_index(self, idx):
		"""Get the absolute index for the list of modules."""
		idx = operator.index(idx)
		if not (-len(self) <= idx < len(self)):
			raise IndexError(f"index {idx} is out of range")
		if idx < 0:
			idx += len(self)
		return str(idx)

	def __getitem__(self, idx: Union[int, slice]) -> Union[Module, "ModuleList"]:
		if isinstance(idx, slice):
			return self.__class__(self._modules[idx])
		else:
			return self._modules[idx]

	def __setitem__(self, idx: int, module: Module) -> None:
		idx = self._get_abs_string_index(idx)
		return setattr(self, str(idx), module)

	def __delitem__(self, idx: Union[int, slice]) -> None:
		if isinstance(idx, slice):
			idxs = []
			for k in range(len(self._modules))[idx]:
				idxs.append(k)
			idxs.reverse()
			for idx in idxs:
				del self._modules[idx]
		else:
			del self._modules[idx]

	def __len__(self) -> int:
		return len(self._modules)

	def __iter__(self) -> Iterator[Module]:
		return iter(self._modules.values())

	def __iadd__(self, modules: Iterable[Module]):
		return self.extend(modules)

	def __add__(self, other: Iterable[Module]):
		combined = ModuleList()
		for i, module in enumerate(chain(self, other)):
			combined.add_module(str(i), module)
		return combined

	def __repr__(self):
		"""Return a custom repr for ModuleList that compresses repeated module representations."""
		list_of_reprs = [repr(item) for item in self]
		if len(list_of_reprs) == 0:
			return self._get_name() + "()"

		start_end_indices = [[0, 0]]
		repeated_blocks = [list_of_reprs[0]]
		for i, r in enumerate(list_of_reprs[1:], 1):
			if r == repeated_blocks[-1]:
				start_end_indices[-1][1] += 1
				continue

			start_end_indices.append([i, i])
			repeated_blocks.append(r)

		lines = []
		main_str = self._get_name() + "("
		for (start_id, end_id), b in zip(start_end_indices, repeated_blocks):
			local_repr = f"({start_id}): {b}"  # default repr

			if start_id != end_id:
				n = end_id - start_id + 1
				local_repr = f"({start_id}-{end_id}): {n} x {b}"

			local_repr = _addindent(local_repr, 2)
			lines.append(local_repr)

		main_str += "\n  " + "\n  ".join(lines) + "\n"
		main_str += ")"
		return main_str

	def __dir__(self):
		keys = super().__dir__()
		keys = [key for key in keys if not key.isdigit()]
		return keys

	def insert(self, index: int, module: Module):
		r"""Insert a given module before a given index in the list.

		Args:
			index (int): index to insert.
			module (nn.Module): module to insert
		"""
		for i in range(len(self._modules), index, -1):
			self._modules[str(i)] = self._modules[str(i - 1)]
		self._modules[str(index)] = module

	def append(self, module: Module):
		r"""Append a given module to the end of the list.

		Args:
			module (nn.Module): module to append
		"""
		self.add_module(str(len(self)), module)
		return self

	def pop(self, key: Union[int, slice]) -> Module:
		v = self[key]
		del self[key]
		return v

	def extend(self, modules: Iterable[Module]):
		r"""Append modules from a Python iterable to the end of the list.

		Args:
			modules (iterable): iterable of modules to append
		"""
		offset = len(self)
		for i, module in enumerate(modules):
			self.add_module(str(offset + i), module)
		return self

	# remove forward alltogether to fallback on Module's _forward_unimplemented
'''
