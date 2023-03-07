import argparse, json
import torch
from models.experimental import attempt_load, End2End
import coremltools as ct

DEFAULT_DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

if __name__ == '__main__':
  parser = argparse.ArgumentParser(
    prog = 'Export model as jit model',
    description = 'Can export a model as a jit model for use in seperate applications.',
    epilog = 'Happy scanning!'
  )

  parser.add_argument('model_path')
  parser.add_argument('-i', '--input_size', type=int, default=640)
  parser.add_argument('-o', '--output_path', default=None)
  parser.add_argument('-c', '--core_ml', default=None)
  parser.add_argument('-d', '--device', default=DEFAULT_DEVICE)

  args = parser.parse_args()

  MODEL_PATH = args.model_path
  SAVE_PATH = '.'.join(MODEL_PATH.split('.')[:-1]) + '.traced.pt'
  if args.output_path != None:
    SAVE_PATH = args.output_path
  DEVICE = args.device

  model = attempt_load(weights=MODEL_PATH, map_location=DEVICE)
  model.eval()

  example_inputs = torch.rand((32, 3, args.input_size, args.input_size))

  traced_model = torch.jit.trace(model, example_inputs=example_inputs)
  torch.jit.save(traced_model, SAVE_PATH)

  if args.core_ml:
    core_ml_model = ct.convert(
      traced_model,
      convert_to="mlprogram",
      inputs=[ct.TensorType(shape=example_inputs.shape)]
    )
    
    core_ml_model.save(args.core_ml)
