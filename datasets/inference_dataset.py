from torch.utils.data import Dataset
from PIL import Image
from utils import data_utils


class InferenceDataset(Dataset):

	def __init__(self, dataset_path, opts, from_transform=None, to_transform = None):
		self.paths = sorted(data_utils.make_dataset(dataset_path))
		self.from_transform = from_transform
		self.to_transform = to_transform
		self.opts = opts

	def __len__(self):
		return len(self.paths)

	def __getitem__(self, index):
		from_path = self.paths[index]
		im = Image.open(from_path)
		im = im.convert('RGB') if self.opts.label_nc == 0 else im.convert('L')

		from_im = self.from_transform(im)
		to_im = self.to_transform(im)
		return from_im, to_im
