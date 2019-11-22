from torch import nn



class CNN(nn.Module):

    def __init__(self):

        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(

            nn.Conv2d(1, 25, kernel_size=3),

            nn.BatchNorm2d(25),

            nn.ReLU(inplace=True)

        )



        self.layer2 = nn.Sequential(

            nn.MaxPool2d(kernel_size=2, stride=2)

        )



        self.layer3 = nn.Sequential(

            nn.Conv2d(25, 50, kernel_size=3),

            nn.BatchNorm2d(50),

            nn.ReLU(inplace=True)

        )



        self.layer4 = nn.Sequential(

            nn.MaxPool2d(kernel_size=2, stride=2)

        )



        self.fc = nn.Sequential(

            nn.Linear(50 * 5 * 5, 1024),

            nn.ReLU(inplace=True),

            nn.Linear(1024, 128),

            nn.ReLU(inplace=True),

            nn.Linear(128, 10)

        )




    def forward(self, x):

        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x = x.view(x.size(0), -1)

        x = self.fc(x)

        return x
