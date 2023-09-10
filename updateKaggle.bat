@echo off
setlocal

set "FolderA=D:\COMPETITIONS\UploadTokaggle"
set "FolderB=TextCompete"
set "FolderC=D:\COMPETITIONS"

rem 步骤1：如果文件夹B存在于文件夹A中，则递归删除它
if exist "%FolderA%\%FolderB%" (
    echo 删除文件夹 %FolderA%\%FolderB%
    rmdir /s /q "%FolderA%\%FolderB%"
)

rem 步骤2：拷贝目录C中的文件夹B到文件夹A中
echo 复制文件夹 %FolderC%\%FolderB% 到 %FolderA%
xcopy /s /i "%FolderC%\%FolderB%" "%FolderA%\%FolderB%"


endlocal