# Ch05

import sys, os
sys.path.append(os.pardir)

from Practice.layer_naiver import *

apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
print(int(price))


dprice=1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_nuim = mul_apple_layer.backward(dapple_price)

print(dapple, dapple_nuim, dtax)

# %%
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orang_layer = AddLayer()
mul_tax_layer = MulLayer()


apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orang_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)

dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orang_layer.backward(dall_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)

print(int(price))
print(int(dapple_num), dapple, int(dorange), dorange_num, dtax)
