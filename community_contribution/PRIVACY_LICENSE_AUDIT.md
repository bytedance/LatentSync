# Privacy and License Audit

## License

The upstream LatentSync repository is Apache License 2.0. Contributions intentionally submitted upstream are expected to be under the same license unless explicitly marked otherwise.

No model weights, compiled TensorRT engines, private video/audio, cache files, or generated benchmark media are included in the proposed public branch.

## Third-Party Code

The clean branch contains original integration code written for this optimization effort. `mohmd21x/LatentSync_trt` was inspected as reference material during local research; prebuilt engines and wholesale copied files are not included.

The TensorRT-RTX runtime remains an external dependency. Users must install it according to NVIDIA/package terms.

## Privacy Scan

The proposed branch was scanned for:

- local Windows drive paths
- Petrus-specific private paths or media labels
- private phone/fish/media terms
- credential-like terms
- model/checkpoint/media artifacts

Machine-specific defaults were removed from the public scripts. Private paths and private media are not included.

## Redistribution Limits

- Do not publish the official model checkpoint or auxiliary model files.
- Do not publish private validation footage.
- Do not publish compiled TensorRT plans as universal artifacts.
- Do not publish local cache directories or benchmark media.

## Remaining Caveat

The public issue and PR may mention that private real-footage validation was performed, but should not include the private footage or filenames.
