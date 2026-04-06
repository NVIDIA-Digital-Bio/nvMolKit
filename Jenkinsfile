@Library('blossom-github-lib@master')
import ipp.blossom.*

// Request any linux node with a GPU; no gpu_type selector so scheduling isn't
// blocked by tainted nodes. restartPolicy/backoffLimit are Pod/Job fields, not container fields.
podTemplate(cloud: 'blsm-prod-cloud', yaml: """
spec:
  containers:
  - name: cuda
    image: nvcr.io/nvidia/cuda:12.6.3-devel-ubuntu22.04
    command:
    - cat
    resources:
      requests:
        nvidia.com/gpu: 1
      limits:
        nvidia.com/gpu: 1
    tty: true
  nodeSelector:
    kubernetes.io/os: linux
""") {
    node(POD_LABEL) {
        timeout(time: 90, unit: 'MINUTES') {
        def githubHelper
        stage('Get Token') {
            withCredentials([usernamePassword(credentialsId: 'github-token', passwordVariable: 'GIT_PASSWORD', usernameVariable: 'GIT_USERNAME')]) {
                githubHelper = GithubHelper.getInstance("${GIT_PASSWORD}", githubData)
            }
        }
        def stageName = ''
        try {
            currentBuild.description = githubHelper.getBuildDescription()
            stageName = 'Verify GPU'
            stage(stageName) {
                container('cuda') {
                    sh 'nvidia-smi'
                }
            }

            stageName = 'Code checkout'
            stage(stageName) {
                githubHelper.updateCommitStatus("$BUILD_URL", "$stageName Running", GitHubCommitState.PENDING)
                if ("Open".equalsIgnoreCase(githubHelper.getPRState())) {
                    println 'PR State is Open'
                    checkout changelog: true, poll: true, scm: [$class: 'GitSCM', branches: [[name: 'pr/' + githubHelper.getPRNumber()]], doGenerateSubmoduleConfigurations: false,
                        submoduleCfg: [],
                        userRemoteConfigs: [[credentialsId: 'github-token', url: githubHelper.getCloneUrl(), refspec: '+refs/pull/*/head:refs/remotes/origin/pr/*']]]
                } else if ("Merged".equalsIgnoreCase(githubHelper.getPRState())) {
                    println 'PR State is Merged'
                    checkout changelog: true, poll: true, scm: [$class: 'GitSCM', branches: [[name: githubHelper.getMergedSHA()]],
                        doGenerateSubmoduleConfigurations: false,
                        submoduleCfg: [],
                        userRemoteConfigs: [[credentialsId: 'github-token', url: githubHelper.getCloneUrl(), refspec: '+refs/pull/*/merge:refs/remotes/origin/pr/*']]]
                }
            }

            stageName = 'Build and Python tests'
            stage(stageName) {
                githubHelper.updateCommitStatus("$BUILD_URL", "$stageName Running", GitHubCommitState.PENDING)
                container('cuda') {
                    sh '''
                        set -ex
                        cd "${WORKSPACE}"
                        export DEBIAN_FRONTEND=noninteractive
                        apt-get update
                        apt-get install -y --no-install-recommends \
                            gcc-12 g++-12 build-essential git wget ca-certificates \
                            libomp-15-dev
                        bash admin/setup_conda_env.sh 3.12 2024.09.6
                        . /usr/local/anaconda/etc/profile.d/conda.sh && conda activate base
                        pip install --upgrade pip
                        pip install torch --index-url https://download.pytorch.org/whl/cu126
                        export CC="$(command -v gcc-12)"
                        export CXX="$(command -v g++-12)"
                        export CMAKE_BUILD_PARALLEL_LEVEL="${CMAKE_BUILD_PARALLEL_LEVEL:-$(nproc)}"
                        export NVMOLKIT_CUDA_TARGET_MODE=native
                        pip install -v .
                        cd nvmolkit/tests
                        pytest -v .
                    '''
                }
                githubHelper.updateCommitStatus("$BUILD_URL", "$stageName Complete", GitHubCommitState.SUCCESS)
            }

            githubHelper.uploadLogs(this, env.JOB_NAME, env.BUILD_NUMBER, null, null)
            githubHelper.updateCommitStatus("$BUILD_URL", 'Complete', GitHubCommitState.SUCCESS)
        } catch (Exception ex) {
            currentBuild.result = 'FAILURE'
            println ex
            githubHelper.uploadLogs(this, env.JOB_NAME, env.BUILD_NUMBER, null, null)
            githubHelper.updateCommitStatus("$BUILD_URL", "$stageName Failed", GitHubCommitState.FAILURE)
        }
        }
    }
}
