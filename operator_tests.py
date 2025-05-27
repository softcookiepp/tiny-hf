
def test_cumprod():
	from tg_adapter import F as tinyF
	inp = np.arange(5*4).reshape(5, 4).astype(np.float32)
	test_function( (inp, 0), torch.cumprod, tinyF.cumprod)


def test_cat():
	a = make_test_data(40, 2, 5)
	b = make_test_data(2, 2, 5)
	test_function( ([a, b], 0), {}, torch.cat, tg_adapter.cat)
