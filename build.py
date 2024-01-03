import os

def __main(main : str):
    parameters = [
        '--mingw64',				# 使用gcc编译器来编译得到的C和C++源文件
        '--standalone',				# 构建独立软件，也就是将于系统有关的运行库和Python运行时打包
        '--show-progress',			# 展示打包过程
        '--show-memory',			# 打印打包时的内存占用
        '--nofollow-imports',		# 不打包import语句导入的包（因为nuitka自动导入的库有问题，后面我们会手动导入，这样成功率更高）
        '--plugin-enable=pylint-warnings',
        '--output-dir=dist'			# 存放构建结果的文件夹
    ]

    param_str = " ".join(parameters)
    command = "nuitka {} {}".format(param_str, main)

    os.system(command)

__main("zhenjuji.py")
