## File structure
```
data/
    raw_data/
        static21067-10-18-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip
        static22846-10-16-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip
        static23343-5-17-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip
        static23656-14-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip
        static23964-4-22-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip
        static26872-17-20-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip
        static27204-5-13-GrayImageNet-94c6ff995dac583098847cfecd43e7b6.zip
    info.yaml
    README.md
```
- `raw_data/` should contain the ZIP files from [gin.g-node.org/cajal/Sensorium2022](https://gin.g-node.org/cajal/Sensorium2022) with their original filenames.
- `info.yaml` is a look up table to keep track of which ZIP file is for Sensorium, Sensoirum+ and pre-training use.
