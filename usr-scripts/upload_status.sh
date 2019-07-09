#!/bin/sh

# branch name
branch=$1

# clone proxy folder and create new branch
git clone --depth=1 --single-branch https://github.com/Luolc/experiment-status-proxy.git ${branch}
cd ${branch} && git checkout -b ${branch}

# copy status file
cp ${branch}.json ${branch}/status-payload/${branch}.json
cd ${branch} && git add status-payload/${branch}.json
cd ${branch} && git commit -m "${branch}"
cd ${branch} && git push \
    https://${GIT_USERNAME}:${GIT_PASSWORD}@github.com/Luolc/experiment-status-proxy.git \
    -u origin ${branch}

# remove proxy folder
rm -rf ${branch}

# remove tmp status file
rm ${branch}.json
