import Header from "@/components/Header";
import Hero from "@/components/Hero";
import DetectionSection from "@/components/detection/DetectionSection";
import HowItWorks from "@/components/HowItWorks";
import Footer from "@/components/Footer";

export default function Home() {
  return (
    <div className="flex min-h-screen flex-col">
      <Header />
      <main>
        <Hero />
        <DetectionSection />
        <HowItWorks />
      </main>
      <Footer />
    </div>
  );
}
