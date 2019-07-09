#!/bin/sh

# branch name
branch=$1

# clone proxy folder and create new branch
git clone --depth=1 --single-branch https://github.com/Luolc/experiment-status-proxy.git ${branch}
cd ${branch}
git checkout -b ${branch}

# copy status file
cp ../${branch}.json status-payload/${branch}.json
git add status-payload/${branch}.json
git commit -m "${branch}"
git push \
    https://${GIT_USERNAME}:${GIT_PASSWORD}@github.com/Luolc/experiment-status-proxy.git \
    -u origin ${branch}

cd ..

# remove proxy folder
rm -rf ${branch}

# remove tmp status file
rm ${branch}.json
