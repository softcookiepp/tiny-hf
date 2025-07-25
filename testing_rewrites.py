XLA_AVAILABLE = False


def retrieve_timesteps(
	scheduler,
	torch,
	num_inference_steps = None,
	device = None,
	timesteps = None,
	sigmas = None,
	**kwargs,
):
	r"""
	Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
	custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

	Args:
		scheduler (`SchedulerMixin`):
			The scheduler to get timesteps from.
		num_inference_steps (`int`):
			The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
			must be `None`.
		device (`str` or `torch.device`, *optional*):
			The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
		timesteps (`List[int]`, *optional*):
			Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
			`num_inference_steps` and `sigmas` must be `None`.
		sigmas (`List[float]`, *optional*):
			Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
			`num_inference_steps` and `timesteps` must be `None`.

	Returns:
		`Tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
		second element is the number of inference steps.
	"""
	if timesteps is not None and sigmas is not None:
		raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
	if timesteps is not None:
		accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
		if not accepts_timesteps:
			raise ValueError(
				f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
				f" timestep schedules. Please check whether you are using the correct scheduler."
			)
		scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
		timesteps = scheduler.timesteps
		num_inference_steps = len(timesteps)
	elif sigmas is not None:
		accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
		if not accept_sigmas:
			raise ValueError(
				f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
				f" sigmas schedules. Please check whether you are using the correct scheduler."
			)
		scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
		timesteps = scheduler.timesteps
		num_inference_steps = len(timesteps)
	else:
		scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
		timesteps = scheduler.timesteps
	return timesteps, num_inference_steps

def sd_pipeline_call(
	self,
	torch,
	prompt = None,
	height = None,
	width = None,
	num_inference_steps = 50,
	timesteps = None,
	sigmas = None,
	guidance_scale = 7.5,
	negative_prompt = None,
	num_images_per_prompt = 1,
	eta: float = 0.0,
	generator = None,
	latents = None,
	prompt_embeds = None,
	negative_prompt_embeds = None,
	ip_adapter_image = None,
	ip_adapter_image_embeds = None,
	output_type = "pil",
	return_dict: bool = False,
	cross_attention_kwargs = None,
	guidance_rescale: float = 0.0,
	clip_skip = None,
	callback_on_step_end = None,
	callback_on_step_end_tensor_inputs = ["latents"],
	**kwargs,
):
	r"""
	The call function to the pipeline for generation.

	Args:
		prompt (`str` or `List[str]`, *optional*):
			The prompt or prompts to guide image generation. If not defined, you need to pass `prompt_embeds`.
		height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
			The height in pixels of the generated image.
		width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
			The width in pixels of the generated image.
		num_inference_steps (`int`, *optional*, defaults to 50):
			The number of denoising steps. More denoising steps usually lead to a higher quality image at the
			expense of slower inference.
		timesteps (`List[int]`, *optional*):
			Custom timesteps to use for the denoising process with schedulers which support a `timesteps` argument
			in their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is
			passed will be used. Must be in descending order.
		sigmas (`List[float]`, *optional*):
			Custom sigmas to use for the denoising process with schedulers which support a `sigmas` argument in
			their `set_timesteps` method. If not defined, the default behavior when `num_inference_steps` is passed
			will be used.
		guidance_scale (`float`, *optional*, defaults to 7.5):
			A higher guidance scale value encourages the model to generate images closely linked to the text
			`prompt` at the expense of lower image quality. Guidance scale is enabled when `guidance_scale > 1`.
		negative_prompt (`str` or `List[str]`, *optional*):
			The prompt or prompts to guide what to not include in image generation. If not defined, you need to
			pass `negative_prompt_embeds` instead. Ignored when not using guidance (`guidance_scale < 1`).
		num_images_per_prompt (`int`, *optional*, defaults to 1):
			The number of images to generate per prompt.
		eta (`float`, *optional*, defaults to 0.0):
			Corresponds to parameter eta (η) from the [DDIM](https://arxiv.org/abs/2010.02502) paper. Only applies
			to the [`~schedulers.DDIMScheduler`], and is ignored in other schedulers.
		generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
			A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
			generation deterministic.
		latents (`torch.Tensor`, *optional*):
			Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
			generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
			tensor is generated by sampling using the supplied random `generator`.
		prompt_embeds (`torch.Tensor`, *optional*):
			Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
			provided, text embeddings are generated from the `prompt` input argument.
		negative_prompt_embeds (`torch.Tensor`, *optional*):
			Pre-generated negative text embeddings. Can be used to easily tweak text inputs (prompt weighting). If
			not provided, `negative_prompt_embeds` are generated from the `negative_prompt` input argument.
		ip_adapter_image: (`PipelineImageInput`, *optional*): Optional image input to work with IP Adapters.
		ip_adapter_image_embeds (`List[torch.Tensor]`, *optional*):
			Pre-generated image embeddings for IP-Adapter. It should be a list of length same as number of
			IP-adapters. Each element should be a tensor of shape `(batch_size, num_images, emb_dim)`. It should
			contain the negative image embedding if `do_classifier_free_guidance` is set to `True`. If not
			provided, embeddings are computed from the `ip_adapter_image` input argument.
		output_type (`str`, *optional*, defaults to `"pil"`):
			The output format of the generated image. Choose between `PIL.Image` or `np.array`.
		return_dict (`bool`, *optional*, defaults to `True`):
			Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
			plain tuple.
		cross_attention_kwargs (`dict`, *optional*):
			A kwargs dictionary that if specified is passed along to the [`AttentionProcessor`] as defined in
			[`self.processor`](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
		guidance_rescale (`float`, *optional*, defaults to 0.0):
			Guidance rescale factor from [Common Diffusion Noise Schedules and Sample Steps are
			Flawed](https://arxiv.org/pdf/2305.08891.pdf). Guidance rescale factor should fix overexposure when
			using zero terminal SNR.
		clip_skip (`int`, *optional*):
			Number of layers to be skipped from CLIP while computing the prompt embeddings. A value of 1 means that
			the output of the pre-final layer will be used for computing the prompt embeddings.
		callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
			A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
			each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
			DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
			list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
		callback_on_step_end_tensor_inputs (`List`, *optional*):
			The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
			will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
			`._callback_tensor_inputs` attribute of your pipeline class.

	Examples:

	Returns:
		[`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
			If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] is returned,
			otherwise a `tuple` is returned where the first element is a list with the generated images and the
			second element is a list of `bool`s indicating whether the corresponding generated image contains
			"not-safe-for-work" (nsfw) content.
	"""

	callback = kwargs.pop("callback", None)
	callback_steps = kwargs.pop("callback_steps", None)

	if callback is not None:
		deprecate(
			"callback",
			"1.0.0",
			"Passing `callback` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
		)
	if callback_steps is not None:
		deprecate(
			"callback_steps",
			"1.0.0",
			"Passing `callback_steps` as an input argument to `__call__` is deprecated, consider using `callback_on_step_end`",
		)
	"""
	if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
		callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs
	"""
	# 0. Default height and width to unet
	if not height or not width:
		height = (
			self.unet.config.sample_size
			if self._is_unet_config_sample_size_int
			else self.unet.config.sample_size[0]
		)
		width = (
			self.unet.config.sample_size
			if self._is_unet_config_sample_size_int
			else self.unet.config.sample_size[1]
		)
		height, width = height * self.vae_scale_factor, width * self.vae_scale_factor
	# to deal with lora scaling and other possible forward hooks

	# 1. Check inputs. Raise error if not correct
	self.check_inputs(
		prompt,
		height,
		width,
		callback_steps,
		negative_prompt,
		prompt_embeds,
		negative_prompt_embeds,
		ip_adapter_image,
		ip_adapter_image_embeds,
		callback_on_step_end_tensor_inputs,
	)

	self._guidance_scale = guidance_scale
	self._guidance_rescale = guidance_rescale
	self._clip_skip = clip_skip
	self._cross_attention_kwargs = cross_attention_kwargs
	self._interrupt = False

	# 2. Define call parameters
	if prompt is not None and isinstance(prompt, str):
		batch_size = 1
	elif prompt is not None and isinstance(prompt, list):
		batch_size = len(prompt)
	else:
		batch_size = prompt_embeds.shape[0]

	device = self._execution_device

	# 3. Encode input prompt
	lora_scale = (
		self.cross_attention_kwargs.get("scale", None) if self.cross_attention_kwargs is not None else None
	)

	prompt_embeds, negative_prompt_embeds = self.encode_prompt(
		prompt,
		device,
		num_images_per_prompt,
		self.do_classifier_free_guidance,
		negative_prompt,
		prompt_embeds=prompt_embeds,
		negative_prompt_embeds=negative_prompt_embeds,
		lora_scale=lora_scale,
		clip_skip=self.clip_skip,
	)

	# For classifier free guidance, we need to do two forward passes.
	# Here we concatenate the unconditional and text embeddings into a single batch
	# to avoid doing two forward passes
	if self.do_classifier_free_guidance:
		prompt_embeds = torch.cat([negative_prompt_embeds, prompt_embeds])

	if ip_adapter_image is not None or ip_adapter_image_embeds is not None:
		image_embeds = self.prepare_ip_adapter_image_embeds(
			ip_adapter_image,
			ip_adapter_image_embeds,
			device,
			batch_size * num_images_per_prompt,
			self.do_classifier_free_guidance,
		)
	# 4. Prepare timesteps
	timesteps, num_inference_steps = retrieve_timesteps(
		self.scheduler, torch, num_inference_steps, device, timesteps, sigmas
	)

	# 5. Prepare latent variables
	num_channels_latents = self.unet.config.in_channels
	latents = self.prepare_latents(
		batch_size * num_images_per_prompt,
		num_channels_latents,
		height,
		width,
		prompt_embeds.dtype,
		device,
		generator,
		latents,
	)
	
	# 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
	extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

	# 6.1 Add image embeds for IP-Adapter
	added_cond_kwargs = (
		{"image_embeds": image_embeds}
		if (ip_adapter_image is not None or ip_adapter_image_embeds is not None)
		else None
	)

	# 6.2 Optionally get Guidance Scale Embedding
	timestep_cond = None
	if self.unet.config.time_cond_proj_dim is not None:
		guidance_scale_tensor = torch.tensor(self.guidance_scale - 1).repeat(batch_size * num_images_per_prompt)
		timestep_cond = self.get_guidance_scale_embedding(
			guidance_scale_tensor, embedding_dim=self.unet.config.time_cond_proj_dim
		).to(device=device, dtype=latents.dtype)
	
	# 7. Denoising loop
	num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
	self._num_timesteps = len(timesteps)
	with self.progress_bar(total=num_inference_steps) as progress_bar:
		for i, t in enumerate(timesteps):
			if self.interrupt:
				continue

			# expand the latents if we are doing classifier free guidance
			latent_model_input = torch.cat([latents] * 2) if self.do_classifier_free_guidance else latents
			
			latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)
			
			# predict the noise residual
			noise_pred = self.unet(
				latent_model_input,
				t,
				encoder_hidden_states=prompt_embeds,
				timestep_cond=timestep_cond,
				cross_attention_kwargs=self.cross_attention_kwargs,
				added_cond_kwargs=added_cond_kwargs,
				return_dict=False,
			)[0]
			
			# perform guidance
			if self.do_classifier_free_guidance:
				noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
				noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_text - noise_pred_uncond)
			
			if self.do_classifier_free_guidance and self.guidance_rescale > 0.0:
				# Based on 3.4. in https://arxiv.org/pdf/2305.08891.pdf
				noise_pred = rescale_noise_cfg(noise_pred, noise_pred_text, guidance_rescale=self.guidance_rescale)

			# compute the previous noisy sample x_t -> x_t-1
			#latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
			ddim_out = ddim_step(self.scheduler, torch, noise_pred, t, latents, **extra_step_kwargs, return_dict=False)
			old_latents = latents
			latents = ddim_out[0]
			
			if i == self._num_timesteps - 1:
				# ddim_out includes the other crap as well
				return noise_pred, t, old_latents, ddim_out
			
			if callback_on_step_end is not None:
				callback_kwargs = {}
				for k in callback_on_step_end_tensor_inputs:
					callback_kwargs[k] = locals()[k]
				callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

				latents = callback_outputs.pop("latents", latents)
				prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
				negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

			# call the callback, if provided
			if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
				progress_bar.update()
				if callback is not None and i % callback_steps == 0:
					step_idx = i // getattr(self.scheduler, "order", 1)
					callback(step_idx, t, latents)
			if XLA_AVAILABLE:
				xm.mark_step()
	
	if not output_type == "latent":
		image = self.vae.decode(latents / self.vae.config.scaling_factor, return_dict=False, generator=generator)[
			0
		]
		image, has_nsfw_concept = self.run_safety_checker(image, device, prompt_embeds.dtype)
	else:
		image = latents
		has_nsfw_concept = None

	if has_nsfw_concept is None:
		do_denormalize = [True] * image.shape[0]
	else:
		do_denormalize = [not has_nsfw for has_nsfw in has_nsfw_concept]
	# skip the processor for now...
	#return image
	image = self.image_processor.postprocess(image, output_type=output_type, do_denormalize=do_denormalize)

	# Offload all models
	self.maybe_free_model_hooks()

	if not return_dict:
		return (image, has_nsfw_concept)

	out = StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
	return out
	
def _get_variance(self, torch, timestep, prev_timestep):
	alpha_prod_t = self.alphas_cumprod[timestep]
	alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod
	beta_prod_t = 1 - alpha_prod_t
	beta_prod_t_prev = 1 - alpha_prod_t_prev

	variance = (beta_prod_t_prev / beta_prod_t) * (1 - alpha_prod_t / alpha_prod_t_prev)

	return variance, alpha_prod_t, alpha_prod_t_prev, beta_prod_t, beta_prod_t_prev
	
def ddim_step(self,
		torch,
		model_output,
		timestep,
		sample,
		eta: float = 0.0,
		use_clipped_model_output: bool = False,
		generator=None,
		variance_noise = None,
		return_dict: bool = True,):
			
	if self.num_inference_steps is None:
		raise ValueError(
			"Number of inference steps is 'None', you need to run 'set_timesteps' after creating the scheduler"
		)

	# See formulas (12) and (16) of DDIM paper https://arxiv.org/pdf/2010.02502.pdf
	# Ideally, read DDIM paper in-detail understanding

	# Notation (<variable name> -> <name in paper>
	# - pred_noise_t -> e_theta(x_t, t)
	# - pred_original_sample -> f_theta(x_t, t) or x_0
	# - std_dev_t -> sigma_t
	# - eta -> η
	# - pred_sample_direction -> "direction pointing to x_t"
	# - pred_prev_sample -> "x_t-1"
	print(type(timestep) )
	# 1. get previous step value (=t-1)
	prev_timestep = timestep - self.config.num_train_timesteps // self.num_inference_steps

	# 2. compute alphas, betas
	alpha_prod_t = self.alphas_cumprod[timestep]
	alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else self.final_alpha_cumprod

	beta_prod_t = 1 - alpha_prod_t

	# 3. compute predicted original sample from predicted noise also called
	# "predicted x_0" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
	if self.config.prediction_type == "epsilon":
		pred_original_sample = (sample - beta_prod_t ** (0.5) * model_output) / alpha_prod_t ** (0.5)
		pred_epsilon = model_output
	elif self.config.prediction_type == "sample":
		pred_original_sample = model_output
		pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
	elif self.config.prediction_type == "v_prediction":
		pred_original_sample = (alpha_prod_t**0.5) * sample - (beta_prod_t**0.5) * model_output
		pred_epsilon = (alpha_prod_t**0.5) * model_output + (beta_prod_t**0.5) * sample
	else:
		raise ValueError(
			f"prediction_type given as {self.config.prediction_type} must be one of `epsilon`, `sample`, or"
			" `v_prediction`"
		)

	# 4. Clip or threshold "predicted x_0"
	if self.config.thresholding:
		pred_original_sample = self._threshold_sample(pred_original_sample)
	elif self.config.clip_sample:
		pred_original_sample = pred_original_sample.clamp(
			-self.config.clip_sample_range, self.config.clip_sample_range
		)

	# 5. compute variance: "sigma_t(η)" -> see formula (16)
	# σ_t = sqrt((1 − α_t−1)/(1 − α_t)) * sqrt(1 − α_t/α_t−1)
	#variance = self._get_variance(timestep, prev_timestep)
	var_out = _get_variance(self, torch, timestep, prev_timestep)
	variance = var_out[0]
	std_dev_t = eta * variance ** (0.5)

	if use_clipped_model_output:
		# the pred_epsilon is always re-derived from the clipped x_0 in Glide
		pred_epsilon = (sample - alpha_prod_t ** (0.5) * pred_original_sample) / beta_prod_t ** (0.5)
	# 6. compute "direction pointing to x_t" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
	pred_sample_direction = (1 - alpha_prod_t_prev - std_dev_t**2) ** (0.5) * pred_epsilon
	# 7. compute x_t without "random noise" of formula (12) from https://arxiv.org/pdf/2010.02502.pdf
	prev_sample = alpha_prod_t_prev ** (0.5) * pred_original_sample + pred_sample_direction
	if eta > 0:
		print(eta)
		if variance_noise is not None and generator is not None:
			raise ValueError(
				"Cannot pass both generator and variance_noise. Please make sure that either `generator` or"
				" `variance_noise` stays `None`."
			)

		if variance_noise is None:
			variance_noise = randn_tensor(
				model_output.shape, generator=generator, device=model_output.device, dtype=model_output.dtype
			)
		# Ok, so the variance is definitely the problem. WHY?
		variance = std_dev_t * variance_noise
		prev_sample = prev_sample + variance
	if not return_dict:
		print(timestep.numpy(), prev_timestep.numpy() )
		return (
			prev_sample,
			pred_original_sample,
			variance_noise,
			std_dev_t,
			timestep,
			prev_timestep,
			var_out
		)
	raise NotImplementedError
	return DDIMSchedulerOutput(prev_sample=prev_sample, pred_original_sample=pred_original_sample)
