Co można spróbować zrobić:
- zwiększyć augmentacje(gaussion noise, brightness, etc)
- zwiększyć ilość kanałów w sieci, zmieniająć ilość warstw konwolucyjnych( 3 conv, 2 full connected + pooling + dropout) -> propozycja prowadzącego
- polepszyć preprocessing danych( równa dystrybucja wśród "left")


na początku:
rgb --> yuv palet
clache (żeby linie wyektrsaktowały)
dodać blur kernel 3,3

augum:
brightness
flipowanie
obracal zdjęcia lustrzane odbicie

model (zmiany):

conv 3,3 64 parametery
max pooling 
conv 32
max pooling
conv 64
max pooling flatten
dense 128
dense 64
dense 32
dense 2 z activ tanh
relu
