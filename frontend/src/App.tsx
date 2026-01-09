import { useState, useEffect } from 'react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Spinner } from '@/components/ui/spinner'
import { toast } from 'sonner'

const API_URL = import.meta.env.VITE_API_URL || 'http://127.0.0.1:8000'

interface Video {
  name: string
  filename: string
  path: string
}

interface StatusData {
  status: string
  progress: number
  message: string
  stage?: string
  output_url?: string
}

interface ProcessedVideo {
  job_id: string
  filename: string
  url: string
  timestamp: number
}

function App() {
  const [availableVideos, setAvailableVideos] = useState<Video[]>([])
  const [selectedVideo, setSelectedVideo] = useState<string | null>(null)
  const [jobId, setJobId] = useState<string | null>(null)
  const [processing, setProcessing] = useState(false)
  const [progress, setProgress] = useState(0)
  const [statusMessage, setStatusMessage] = useState('')
  const [outputUrl, setOutputUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [processedVideos, setProcessedVideos] = useState<ProcessedVideo[]>([])
  const [historyLoading, setHistoryLoading] = useState(false)

  useEffect(() => {
    const fetchVideos = async () => {
      try {
        const response = await fetch(`${API_URL}/available-videos`)
        const data = await response.json()

        if (data.status === 'ok') {
          setAvailableVideos(data.videos)
        } else {
          toast.error('Failed to load videos: ' + data.message)
        }
      } catch (error) {
        toast.error('Could not fetch video list: ' + error)
      } finally {
        setLoading(false)
      }
    }

    fetchVideos()
  }, [])

  useEffect(() => {
    const fetchHistory = async () => {
      setHistoryLoading(true)
      try {
        const response = await fetch(`${API_URL}/videos`)
        const data = await response.json()
        setProcessedVideos(data.videos || [])
      } catch (error) {
        console.error('Failed to load history:', error)
      } finally {
        setHistoryLoading(false)
      }
    }

    fetchHistory()
  }, [outputUrl])

  const startProcessing = async () => {
    if (!selectedVideo) return

    setProcessing(true)
    setOutputUrl(null)

    try {
      const response = await fetch(`${API_URL}/process?video_name=${selectedVideo}`, {
        method: 'POST',
      })
      const data = await response.json()

      if (data.status === 'ok') {
        const newJobId = data.job_id
        setJobId(newJobId)
        toast.success('Video processing started!')

        const interval = setInterval(async () => {
          const statusResponse = await fetch(`${API_URL}/status/${newJobId}`)
          const statusData: StatusData = await statusResponse.json()

          setProgress(statusData.progress || 0)
          setStatusMessage(statusData.message || '')

          if (statusData.status === 'completed') {
            clearInterval(interval)
            setProcessing(false)
            setOutputUrl(statusData.output_url || null)
            toast.success('Video processing completed!')
          } else if (statusData.status === 'error') {
            clearInterval(interval)
            setProcessing(false)
            toast.error('Error: ' + statusData.message)
          }
        }, 1000)
      } else {
        toast.error(data.message || 'Failed to start processing')
        setProcessing(false)
      }
    } catch (error) {
      toast.error('Failed to start processing: ' + error)
      setProcessing(false)
    }
  }

  const reset = () => {
    window.location.reload()
  }

  const formatTimestamp = (timestamp: number) => {
    const date = new Date(timestamp * 1000)
    return date.toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      hour: '2-digit',
      minute: '2-digit'
    })
  }

  return (
    <div className="min-h-screen bg-background p-4 md:p-8">
      <div className="mx-auto max-w-6xl space-y-8">
        <div className="text-center space-y-2">
          <h1 className="text-4xl font-bold tracking-tight text-foreground">Basketball DeepStream</h1>
        </div>

        <div className="space-y-6">
          <h2 className="text-2xl font-semibold text-center text-foreground">
            Select a video to process
          </h2>

          {loading ? (
            <div className="text-center py-12 text-muted-foreground">
              Loading videos...
            </div>
          ) : (
            <>
              <div className={`grid grid-cols-1 md:grid-cols-3 gap-6 transition-all duration-500 ${
                outputUrl ? 'opacity-0 h-0 overflow-hidden' : 'opacity-100'
              }`}>
                {availableVideos.map((video) => (
                  <Card
                    key={video.name}
                    className={`cursor-pointer transition-all hover:shadow-lg ${
                      selectedVideo === video.name
                        ? 'ring-2 ring-primary border-primary shadow-xl'
                        : 'hover:border-primary/30'
                    } ${processing ? 'opacity-50 cursor-not-allowed' : ''}`}
                    onClick={() => !processing && setSelectedVideo(video.name)}
                  >
                    <div className="relative">
                      <video
                        key={selectedVideo === video.name ? 'playing' : 'paused'}
                        src={`${API_URL}${video.path}`}
                        className="w-full h-48 object-cover rounded-t-lg bg-black"
                        muted
                        autoPlay={selectedVideo === video.name}
                        playsInline
                      />
                      {selectedVideo === video.name && (
                        <div className="absolute top-2 right-2 bg-primary text-primary-foreground px-3 py-1.5 rounded-lg text-sm font-semibold shadow-md">
                          ✓ Selected
                        </div>
                      )}
                    </div>
                    <CardContent className="p-2">
                      <h3 className="text-base font-semibold text-center text-card-foreground">
                        {video.name === 'video_1'
                          ? 'Video 1'
                          : video.name === 'video_2'
                          ? 'Video 2'
                          : video.name === 'video_3'
                          ? 'Video 3'
                          : video.name}
                      </h3>
                    </CardContent>
                  </Card>
                ))}
              </div>

              <div className={`text-center pt-4 transition-all duration-500 ${
                outputUrl ? 'opacity-0 h-0 overflow-hidden' : 'opacity-100'
              }`}>
                <Button
                  onClick={startProcessing}
                  disabled={!selectedVideo || processing}
                  size="lg"
                  variant="default"
                  className="text-xl font-bold px-16 py-6 shadow-xl hover:shadow-2xl hover:scale-105 transition-all duration-200 disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {processing ? <Spinner className="size-6" /> : 'Process Video'}
                </Button>
              </div>
            </>
          )}
        </div>

        {processing && (
          <Card className="border-primary/50 shadow-xl max-w-2xl mx-auto animate-in fade-in duration-700">
            <CardHeader className="pb-3">
              <CardTitle className="text-center text-foreground text-2xl">Processing Video</CardTitle>
            </CardHeader>
            <CardContent className="space-y-4">
              <div className="text-center">
                <p
                  key={statusMessage}
                  className="text-xl font-semibold text-foreground animate-in fade-in duration-500"
                >
                  {statusMessage}
                </p>
              </div>

              <div className="space-y-2">
                <Progress value={progress} className="h-3 transition-all duration-300" />
                <div className="text-center">
                  <span className="text-2xl font-bold text-primary transition-all duration-300">
                    {progress}%
                  </span>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        {outputUrl && (
          <Card className="border-primary/50 shadow-lg animate-in fade-in duration-700">
            <CardHeader>
              <CardTitle className="text-foreground text-2xl text-center">Processed Video</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <video
                src={`${API_URL}${outputUrl}`}
                controls
                className="w-full rounded-lg bg-black shadow-md"
              />
              <div className="flex justify-center gap-4">
                <Button
                  onClick={() => window.open(`${API_URL}/video/${jobId}`, '_blank')}
                  variant="default"
                  size="lg"
                  className="text-lg font-semibold px-12 py-6 shadow-lg hover:shadow-xl transition-all"
                >
                  Download
                </Button>
                <Button
                  onClick={reset}
                  variant="outline"
                  size="lg"
                  className="text-lg font-semibold px-12 py-6 shadow-md hover:shadow-lg transition-all"
                >
                  Select New Video
                </Button>
              </div>
            </CardContent>
          </Card>
        )}

        {processedVideos.length > 0 && (
          <Card className="border-muted shadow-md mt-12">
            <CardHeader>
              <CardTitle className="text-foreground text-xl">Processing History</CardTitle>
            </CardHeader>
            <CardContent>
              {historyLoading ? (
                <div className="text-center py-8 text-muted-foreground">Loading history...</div>
              ) : (
                <div className="grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4">
                  {processedVideos.map((video) => (
                    <Card
                      key={video.job_id}
                      className="cursor-pointer hover:shadow-lg transition-all hover:border-primary/30"
                      onClick={() => window.open(`${API_URL}${video.url}`, '_blank')}
                    >
                      <div className="relative aspect-video bg-black rounded-t-lg overflow-hidden">
                        <video
                          src={`${API_URL}${video.url}`}
                          className="w-full h-full object-cover"
                          muted
                          preload="metadata"
                        />
                        <div className="absolute inset-0 bg-black/40 flex items-center justify-center opacity-0 hover:opacity-100 transition-opacity">
                          <span className="text-white text-sm font-semibold">▶ Play</span>
                        </div>
                      </div>
                      <CardContent className="p-2">
                        <p className="text-xs text-muted-foreground text-center">
                          {formatTimestamp(video.timestamp)}
                        </p>
                      </CardContent>
                    </Card>
                  ))}
                </div>
              )}
            </CardContent>
          </Card>
        )}
      </div>
    </div>
  )
}

export default App
