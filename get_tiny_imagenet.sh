#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")" || exit 1
DEST="${1:-./data}"
mkdir -p "$DEST"
if [ -f "$DEST/tiny-imagenet-200/wnids.txt" ]; then
  echo "already there: $DEST/tiny-imagenet-200"; exit 0
fi
Z="$DEST/tiny-imagenet-200.zip"
for url in https://cs231n.stanford.edu/tiny-imagenet-200.zip http://cs231n.stanford.edu/tiny-imagenet-200.zip; do
  if wget -q --tries=3 --timeout=60 -O "$Z" "$url" && unzip -tq "$Z" >/dev/null 2>&1; then
    unzip -q -o "$Z" -d "$DEST" && rm -f "$Z"
    break
  fi
  echo "download failed from $url"
done
rm -f "$Z"
if [ ! -f "$DEST/tiny-imagenet-200/wnids.txt" ] || [ ! -f "$DEST/tiny-imagenet-200/val/val_annotations.txt" ]; then
  echo "Tiny-ImageNet-200 could not be downloaded. Get tiny-imagenet-200.zip by hand and unzip it into $DEST."
  exit 1
fi
echo "Tiny-ImageNet-200 ready: $DEST/tiny-imagenet-200"
